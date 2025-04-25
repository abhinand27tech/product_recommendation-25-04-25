import React, { useState, useEffect } from 'react';
import { Container, Box, Typography, Alert, CircularProgress, Button, Grid, Paper } from '@mui/material';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { SOURCE_COLORS } from '../constants/colors';

const RecommendationPage = ({ selectedCustomerId, selectedGodownCode }) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingProductNames, setLoadingProductNames] = useState(false);
  const [error, setError] = useState(null);
  const [productNames, setProductNames] = useState({});

  const fetchProductNames = async () => {
    try {
      setLoadingProductNames(true);
      const response = await fetch('http://localhost:5000/api/product-names');
      const data = await response.json();
      if (response.ok) {
        setProductNames(data);
      } else {
        throw new Error(data.error || 'Failed to fetch product names');
      }
    } catch (err) {
      console.error('Error fetching product names:', err);
      setError('Failed to load product information');
    } finally {
      setLoadingProductNames(false);
    }
  };

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);

      // Validate inputs
      if (!selectedCustomerId || !selectedGodownCode) {
        setError('Please select both customer and godown');
        return;
      }

      const response = await fetch(
        `http://localhost:5000/api/recommendations/${selectedCustomerId}?godown_code=${encodeURIComponent(selectedGodownCode)}`
      );
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch recommendations');
      }

      // Validate response format
      if (!Array.isArray(data.recommendations)) {
        throw new Error('Invalid response format from server');
      }

      console.log('Received recommendations:', data.recommendations);
      setRecommendations(data.recommendations);
      
      if (data.recommendations.length === 0) {
        setError('No recommendations available for this customer');
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch recommendations. Please try again later.');
      console.error('Error fetching recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProductNames();
  }, []);

  useEffect(() => {
    if (selectedCustomerId && selectedGodownCode) {
      fetchRecommendations();
    }
  }, [selectedCustomerId, selectedGodownCode]);

  const RecommendationLegend = () => (
    <Paper elevation={2} sx={{ p: 2, mb: 3, backgroundColor: '#f8f9fa' }}>
      <Typography variant="h6" gutterBottom>
        Recommendation Methods
      </Typography>
      <Grid container spacing={2}>
        {Object.entries(SOURCE_COLORS).map(([key, color]) => (
          key !== 'default' && (
            <Grid item xs={12} sm={6} md={4} key={key}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 20, height: 20, backgroundColor: color, borderRadius: 1 }} />
                <Typography variant="body2">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</Typography>
              </Box>
            </Grid>
          )
        ))}
      </Grid>
    </Paper>
  );

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Product Recommendations
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {(loading || loadingProductNames) ? (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            minHeight: '300px',
            gap: 2
          }}>
            <CircularProgress size={60} />
            <Typography variant="body1" color="text.secondary">
              {loadingProductNames ? 'Loading product information...' : 'Generating personalized recommendations...'}
            </Typography>
          </Box>
        ) : (
          <>
            <RecommendationLegend />
            <RecommendationDisplay 
              recommendations={recommendations}
              productNames={productNames}
            />
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
              <Button 
                variant="contained" 
                onClick={fetchRecommendations}
                disabled={loading || !selectedCustomerId || !selectedGodownCode}
                sx={{ minWidth: '200px' }}
              >
                Refresh Recommendations
              </Button>
            </Box>
          </>
        )}
      </Box>
    </Container>
  );
};

export default RecommendationPage; 