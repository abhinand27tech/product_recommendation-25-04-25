import React, { useState } from 'react';
import { useProductNames } from './ProductNameContext';
import { SOURCE_COLORS, SOURCE_WEIGHTS, SOURCE_LABELS } from '../constants/colors';
import './styles.css';

function Recommendations({ customerId, godownCode }) {
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { productNames, loading: namesLoading } = useProductNames();

    const getProductName = (itemNo) => {
        const id = String(itemNo).trim();
        return productNames[id] || `Unknown Product (${id})`;
    };

    const getSourceTypes = (sources) => {
        return sources.map(source => {
            const sourceKey = source.trim();
            return {
                text: SOURCE_LABELS[sourceKey] || SOURCE_LABELS.default,
                color: SOURCE_COLORS[sourceKey] || SOURCE_COLORS.default
            };
        });
    };

    // Restore previous confidence calculation based on position
    const calculateConfidence = (index) => {
        const baseConfidence = 98;  // Start with 98% confidence
        const decrease = index * 2;  // Decrease by 2% for each position
        return Math.max(baseConfidence - decrease, 70);  // Minimum 70%
    };

    const fetchRecommendations = async () => {
        if (!customerId) {
            setError('Please select a customer first');
            return;
        }

        if (!godownCode) {
            setError('Please select a godown first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            console.log('Fetching recommendations:', {
                customerId,
                godownCode,
                timestamp: new Date().toISOString()
            });

            const response = await fetch(`http://localhost:5000/api/recommendations?customer_id=${customerId}&godown_code=${godownCode}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch recommendations');
            }

            console.log('Raw recommendations response:', data);

            // Process and analyze recommendations
            const recommendations = data.recommendations || [];
            
            // Log source distribution
            const sourcesDistribution = recommendations.reduce((acc, [_, sources]) => {
                sources.forEach(source => {
                    acc[source] = (acc[source] || 0) + 1;
                });
                return acc;
            }, {});
            
            console.log('Recommendation sources distribution:', sourcesDistribution);

            // Analyze demographic-based recommendations
            const demographicRecs = recommendations.filter(([_, sources]) => 
                sources.includes('gender') || sources.includes('age')
            );

            console.log('Demographic-based recommendations:', {
                total: demographicRecs.length,
                items: demographicRecs.map(([itemNo, sources]) => ({
                    productId: itemNo,
                    productName: getProductName(itemNo),
                    sources: sources
                }))
            });

            setRecommendations(recommendations);
            
            if (recommendations.length === 0) {
                setError('No recommendations available for this customer');
            }
        } catch (err) {
            console.error('Error fetching recommendations:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="recommendations-section">
            <div className="section-header">
                <h2>Product Recommendations</h2>
                <button 
                    onClick={fetchRecommendations}
                    disabled={!customerId || !godownCode || loading || namesLoading}
                    className="fetch-button"
                >
                    {loading || namesLoading ? 'Loading...' : 'Get Recommendations'}
                </button>
            </div>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {(loading || namesLoading) && (
                <div className="loading">
                    <div className="loading-spinner"></div>
                </div>
            )}

            {!loading && !namesLoading && !error && recommendations.length > 0 && (
                <div className="recommendations-container">
                    {recommendations.map(([itemId, sources], index) => {
                        const productName = getProductName(itemId);
                        const sourceTypes = getSourceTypes(sources);
                        const confidence = calculateConfidence(index);
                        
                        return (
                            <div key={itemId} className="recommendation-card">
                                <span className="recommendation-number">#{index + 1}</span>
                                <div className="recommendation-content">
                                    <h3>{productName}</h3>
                                    <div className="product-id">Product ID: {itemId}</div>
                                    <div className="recommendation-sources">
                                        {sourceTypes.map((source, i) => (
                                            <span 
                                                key={i} 
                                                className="source-chip"
                                                style={{ backgroundColor: source.color }}
                                            >
                                                {source.text}
                                            </span>
                                        ))}
                                    </div>
                                    <div className="confidence-info">
                                        <span className="confidence-label">Confidence Score:</span>
                                        <span className="confidence-value">{confidence}%</span>
                                    </div>
                                    <div className="recommendation-score">
                                        <div 
                                            className="score-bar" 
                                            style={{ width: `${confidence}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {!loading && !namesLoading && !error && recommendations.length === 0 && customerId && godownCode && (
                <div className="no-recommendations">
                    No recommendations available for this customer. Try selecting a different customer or godown.
                </div>
            )}
        </div>
    );
}

export default Recommendations; 