from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_services import RecommendationSystem
import os
import threading
import atexit
import time
import pandas as pd

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables to track initialization state
initialization_complete = False
initialization_error = None
initialization_progress = "Starting initialization..."
initialization_lock = threading.Lock()
init_thread = None
start_time = None

# Initialize recommendation system
recommendation_system = RecommendationSystem()

def update_progress(message):
    global initialization_progress
    initialization_progress = message
    print(f"Progress: {message}")

def initialize_system():
    global initialization_complete, initialization_error, initialization_progress, recommendation_system, start_time
    
    # Use a lock to prevent multiple initializations
    if initialization_lock.locked():
        print("Initialization already in progress")
        return
        
    with initialization_lock:
        if initialization_complete:
            print("System already initialized")
            return
            
        try:
            start_time = time.time()
            update_progress("Checking data files...")
            
            # Define data paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            detail_path = os.path.join(current_dir, 'data', 'Detail_comb.csv')
            header_path = os.path.join(current_dir, 'data', 'Header_comb.csv')

            # Check if data files exist
            if not os.path.exists(detail_path):
                raise FileNotFoundError(f"Detail file not found at {detail_path}")
            if not os.path.exists(header_path):
                raise FileNotFoundError(f"Header file not found at {header_path}")

            update_progress("Loading header file...")
            print(f"\nLooking for data files in: {os.path.join(current_dir, 'data')}")
            print(f"Detail file path: {detail_path}")
            print(f"Header file path: {header_path}")

            # Load data
            data_loaded = recommendation_system.load_data(
                detail_path, 
                header_path,
                progress_callback=update_progress
            )
            
            if not data_loaded:
                raise Exception("Failed to load data")
            
            elapsed_time = time.time() - start_time
            initialization_complete = True
            update_progress(f"Initialization complete! (Took {elapsed_time:.1f} seconds)")
            print("Data loaded successfully")
            
        except Exception as e:
            initialization_error = str(e)
            initialization_progress = f"Error: {str(e)}"
            print(f"Error during initialization: {str(e)}")

def cleanup_on_exit():
    """Cleanup function to run when the server shuts down"""
    global init_thread
    if init_thread and init_thread.is_alive():
        print("Shutting down initialization thread...")
        init_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to finish

atexit.register(cleanup_on_exit)

@app.route('/api/status')
def get_status():
    """Get the current initialization status"""
    global start_time
    elapsed = time.time() - start_time if start_time else 0
    return jsonify({
        'initialized': initialization_complete,
        'error': initialization_error,
        'progress': initialization_progress,
        'elapsed_seconds': int(elapsed)
    })

@app.route('/api/godowns', methods=['GET'])
def get_godowns():
    """Get list of all godowns"""
    try:
        if not initialization_complete:
            return jsonify({
                'success': False,
                'error': 'System is still initializing',
                'progress': initialization_progress
            }), 503

        if initialization_error:
            return jsonify({
                'success': False,
                'error': f'System failed to initialize: {initialization_error}'
            }), 500

        godowns = recommendation_system.get_godowns()
        print(f"Retrieved godowns: {godowns}")  # Debug log
        
        if not godowns:
            return jsonify({
                'success': False,
                'error': 'No godowns found'
            }), 404
            
        return jsonify({
            'success': True,
            'godowns': godowns
        })
    except Exception as e:
        print(f"Error getting godowns: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/customers', methods=['POST'])
def get_customers():
    """Get customers for a given godown code"""
    try:
        if not initialization_complete:
            return jsonify({
                'success': False,
                'error': 'System is still initializing',
                'progress': initialization_progress
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        godown_code = data.get('godown_code')
        if not godown_code:
            return jsonify({
                'success': False,
                'error': 'Godown code is required'
            }), 400
        
        print(f"Fetching customers for godown: {godown_code}")  # Debug log
        customers = recommendation_system.get_customers_by_godown(godown_code)
        
        if not customers:
            return jsonify({
                'success': False,
                'error': 'No customers found for this godown'
            }), 404
            
        print(f"Found {len(customers)} customers")  # Debug log
        return jsonify({
            'success': True,
            'customers': customers
        })
    except Exception as e:
        print(f"Error getting customers: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations', methods=['GET'])
@app.route('/api/recommendations/<customer_id>', methods=['GET'])
def get_recommendations(customer_id=None):
    try:
        # Get customer_id either from URL parameter or query parameter
        if customer_id is None:
            customer_id = request.args.get('customer_id')
        
        # Get godown_code from query parameters
        godown_code = request.args.get('godown_code')
        
        if not customer_id or not godown_code:
            return jsonify({
                'error': 'Missing required parameters. Both customer_id and godown_code are required.'
            }), 400
        
        print(f"Generating recommendations for customer {customer_id} in godown {godown_code}")
        
        recommendations = recommendation_system.get_recommendations(
            customer_id, 
            n_recommendations=50,  # Increased to 50 recommendations
            godown_code=godown_code
        )
        
        if not recommendations:
            return jsonify({
                'recommendations': [],
                'message': 'No recommendations available for this customer'
            })
        
        # Format recommendations to include all sources
        formatted_recommendations = []
        for item, sources in recommendations:
            formatted_recommendations.append([item, sources])
        
        print(f"Generated {len(recommendations)} recommendations")
        print("Recommendation format:", formatted_recommendations[:2])  # Print first 2 for debugging
        
        return jsonify({
            'recommendations': formatted_recommendations,
            'message': f'Found {len(recommendations)} recommendations'
        })
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/product-names', methods=['GET'])
def get_product_names():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        product_names_path = os.path.join(current_dir, 'data', 'item_no_product_names1.csv')
        
        if not os.path.exists(product_names_path):
            return jsonify({'error': 'Product names file not found'}), 404
            
        # Read the CSV file with the required columns
        df = pd.read_csv(product_names_path, 
                        usecols=['item_no', 'item_name', 'printable_name'],
                        dtype={'item_no': str, 'item_name': str, 'printable_name': str})
        
        # Create a dictionary mapping item_no to its name
        product_names = {}
        for _, row in df.iterrows():
            item_no = str(row['item_no']).strip()
            # Try printable_name first, if it's empty or NaN, use item_name
            name = row['printable_name']
            if pd.isna(name) or str(name).strip() == '':
                name = row['item_name']
            
            name = str(name).strip()
            if name and name.lower() != 'nan' and item_no:
                product_names[item_no] = name
        
        print(f"Loaded {len(product_names)} product names")
        # Print first few entries to verify format
        first_few = dict(list(product_names.items())[:5])
        print(f"Sample product names: {first_few}")
        
        return jsonify({'product_names': product_names})
    except Exception as e:
        print(f"Error in get_product_names: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nStarting Recommendation System Server...")
    print("Note: Initial data loading may take several minutes.")
    print("Progress updates will be shown here and in the web interface.")
    
    # Start initialization in a background thread only if not already running
    if not initialization_complete and (not init_thread or not init_thread.is_alive()):
        init_thread = threading.Thread(target=initialize_system)
        init_thread.start()
    
    # Start the Flask server without debug mode to prevent double initialization
    app.run(host='0.0.0.0', port=5000) 