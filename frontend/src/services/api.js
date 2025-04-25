import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export const getRecommendations = async (customerId, godownCode) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/recommendations/${customerId}`, {
      params: {
        godown_code: godownCode
      }
    });
    
    if (response.data.error) {
      throw new Error(response.data.error);
    }
    
    // Ensure recommendations are in the correct format
    const recommendations = response.data.recommendations || [];
    console.log('Raw recommendations:', recommendations);
    
    return recommendations;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
};

export const getProductNames = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/product-names`);
    
    if (response.data.error) {
      throw new Error(response.data.error);
    }
    
    // Log the response for debugging
    console.log('Product names API response:', response.data);
    
    // Return the product_names object directly
    return response.data;
  } catch (error) {
    console.error('Error fetching product names:', error);
    throw error;
  }
}; 