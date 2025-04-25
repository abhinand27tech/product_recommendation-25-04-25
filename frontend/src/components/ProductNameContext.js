import React, { createContext, useState, useContext, useEffect } from 'react';

const ProductNameContext = createContext({});

export function ProductNameProvider({ children }) {
    const [productNames, setProductNames] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadProductNames = async () => {
            try {
                console.log("Starting to load product names...");
                const response = await fetch('http://localhost:5000/api/product-names');
                
                if (!response.ok) {
                    throw new Error('Failed to fetch product names');
                }
                
                const data = await response.json();
                
                if (data.product_names) {
                    console.log("Received product names data. Sample:", 
                        Object.entries(data.product_names).slice(0, 5));
                    setProductNames(data.product_names);
                } else {
                    throw new Error('Invalid product names data format');
                }
                
                setLoading(false);
            } catch (err) {
                console.error('Error loading product names:', err);
                setError(err.message);
                setLoading(false);
            }
        };

        loadProductNames();
    }, []);

    const value = {
        productNames,
        loading,
        error,
        getProductName: (itemNo) => {
            const id = String(itemNo).trim();
            return productNames[id] || `Unknown Product (${id})`;
        }
    };

    return (
        <ProductNameContext.Provider value={value}>
            {children}
        </ProductNameContext.Provider>
    );
}

export function useProductNames() {
    return useContext(ProductNameContext);
} 