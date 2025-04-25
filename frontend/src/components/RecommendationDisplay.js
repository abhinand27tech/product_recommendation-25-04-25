import React from 'react';
import { Box, Typography, List, ListItem, ListItemText, Paper, Tooltip, Grid, Chip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import { SOURCE_COLORS, SOURCE_LABELS, SOURCE_WEIGHTS } from '../constants/colors';

const RecommendationDisplay = ({ recommendations, productNames }) => {
  // Group recommendations by source
  const groupRecommendations = () => {
    const groups = Object.keys(SOURCE_WEIGHTS).reduce((acc, key) => {
      acc[key] = [];
      return acc;
    }, {});

    recommendations.forEach(([itemId, sources]) => {
      sources.forEach(source => {
        const sourceKey = source.trim();
        if (groups[sourceKey]) {
          groups[sourceKey].push(itemId);
        }
      });
    });

    return groups;
  };

  const getSourceTitle = (source) => {
    return SOURCE_LABELS[source] || SOURCE_LABELS.default;
  };

  const getSourceColor = (source) => {
    return SOURCE_COLORS[source] || SOURCE_COLORS.default;
  };

  const getProductDisplay = (itemId) => {
    // Convert itemId to string to ensure consistent lookup
    const id = String(itemId).trim();
    const productName = productNames[id];
    
    if (!productName) {
      console.log(`No product name found for ID: ${id}`);
      return id;
    }
    
    // Clean up the product name
    const cleanName = productName.trim();
    // If name is too long, truncate it
    if (cleanName.length > 40) {
      return `${cleanName.substring(0, 37)}...`;
    }
    return cleanName;
  };

  const groups = groupRecommendations();

  return (
    <Box>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        p: 2,
        bgcolor: '#F8F9FA',
        borderRadius: 1,
        mb: 3
      }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold', color: '#2C3E50' }}>
          Product Recommendations
        </Typography>
        <Tooltip title="View recommendations for this customer">
          <button className="btn btn-primary">
            Get Recommendations
          </button>
        </Tooltip>
      </Box>

      {Object.entries(groups).map(([source, items]) => items.length > 0 && (
        <Paper key={source} sx={{ mb: 2, p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6" sx={{ color: getSourceColor(source), flex: 1 }}>
              {getSourceTitle(source)}
            </Typography>
            <Tooltip title={`Recommendations based on ${getSourceTitle(source).toLowerCase()} analysis`}>
              <InfoIcon color="action" sx={{ ml: 1 }} />
            </Tooltip>
          </Box>
          <List dense>
            {items.map((itemId) => (
              <ListItem key={itemId}>
                <ListItemText
                  primary={getProductDisplay(itemId)}
                  secondary={`ID: ${itemId}`}
                />
                <Chip
                  label={`Weight: ${(SOURCE_WEIGHTS[source] * 100).toFixed(0)}%`}
                  size="small"
                  sx={{
                    backgroundColor: getSourceColor(source),
                    color: 'white',
                    ml: 1
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      ))}
    </Box>
  );
};

export default RecommendationDisplay; 