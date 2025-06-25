
import React, { useState } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const checkPhishing = async () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://127.0.0.1:2002/check_phishing', {
        url: url.trim()
      });
      
      setResult(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to check URL. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    checkPhishing();
  };

  const clearInput = () => {
    setUrl('');
    setResult(null);
    setError('');
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%)',
      color: 'white',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Navigation */}
      <nav style={{
        position: 'fixed',
        top: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '100%',
        backgroundColor: '#333',
        padding: '14px 0',
        textAlign: 'center',
        zIndex: 1000
      }}>
        <div style={{ display: 'inline-flex', gap: '20px' }}>
          <a href="/" style={{ color: 'white', textDecoration: 'none', padding: '14px 16px', borderRadius: '10px' }}>
            HOME
          </a>
          <a href="/about" style={{ color: 'white', textDecoration: 'none', padding: '14px 16px', borderRadius: '10px' }}>
            ABOUT
          </a>
          <a href="/contact" style={{ color: 'white', textDecoration: 'none', padding: '14px 16px', borderRadius: '10px' }}>
            CONTACT
          </a>
        </div>
      </nav>

      {/* Main Content */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        paddingTop: '80px',
        textAlign: 'center'
      }}>
        <h1 style={{
          fontSize: '50px',
          fontWeight: 'bold',
          color: '#fff',
          marginBottom: '40px',
          letterSpacing: '3px'
        }}>
          SECURE-SURF
        </h1>
        
        <p style={{
          fontSize: '20px',
          marginBottom: '40px',
          color: '#ccc'
        }}>
          Advanced Phishing Detection System
        </p>

        {/* Input Form */}
        <form onSubmit={handleSubmit} style={{ marginBottom: '40px' }}>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter URL to check..."
              style={{
                padding: '10px 40px 10px 10px',
                border: '1px solid #fff',
                borderRadius: '15px',
                color: '#fff',
                height: '35px',
                width: '425px',
                fontSize: '16px',
                fontWeight: 'bold',
                backgroundColor: 'transparent',
                letterSpacing: '2px'
              }}
            />
            {url && (
              <span
                onClick={clearInput}
                style={{
                  cursor: 'pointer',
                  position: 'absolute',
                  right: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  fontSize: '20px',
                  fontWeight: 'bold',
                  color: 'rgba(255, 255, 255, 0.986)'
                }}
              >
                ×
              </span>
            )}
          </div>
          
          <button
            type="submit"
            disabled={loading}
            style={{
              backgroundColor: 'white',
              color: '#2976d3',
              border: '1px solid #2976d3',
              height: '45px',
              width: '100px',
              borderRadius: '8px',
              fontWeight: 'bold',
              fontSize: '105%',
              cursor: loading ? 'not-allowed' : 'pointer',
              marginLeft: '20px',
              transition: 'all 0.3s ease'
            }}
          >
            {loading ? 'Scanning...' : 'SCAN'}
          </button>
        </form>

        {/* Error Message */}
        {error && (
          <div style={{
            color: '#ff4444',
            fontSize: '18px',
            marginBottom: '20px',
            padding: '10px',
            backgroundColor: 'rgba(255, 68, 68, 0.1)',
            borderRadius: '8px'
          }}>
            {error}
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div style={{
            marginTop: '40px',
            width: '80%',
            maxWidth: '800px'
          }}>
            {/* Main Result */}
            <div style={{
              backgroundColor: result.result_text?.includes('phishing') ? '#ff4444' : '#44ff44',
              color: 'black',
              padding: '20px',
              borderRadius: '15px',
              fontSize: '24px',
              fontWeight: 'bold',
              marginBottom: '30px'
            }}>
              {result.result_text || 'Analysis complete'}
            </div>

            {/* Additional Info Table */}
            {result.additional_info && Object.keys(result.additional_info).length > 0 && (
              <div style={{
                background: 'linear-gradient(170deg, rgba(43, 86, 244, 0.8), rgba(219, 81, 196, 0.8))',
                padding: '20px',
                borderRadius: '15px'
              }}>
                <h3 style={{
                  color: 'white',
                  marginBottom: '20px',
                  fontSize: '25px'
                }}>
                  Detailed Analysis
                </h3>
                
                <table style={{
                  width: '100%',
                  borderCollapse: 'separate',
                  borderSpacing: '0',
                  borderRadius: '18px',
                  overflow: 'hidden',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)'
                }}>
                  <thead>
                    <tr style={{ backgroundColor: '#5d6dff' }}>
                      <th style={{ padding: '12px', textAlign: 'left', color: 'white' }}>Property</th>
                      <th style={{ padding: '12px', textAlign: 'left', color: 'white' }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.additional_info).map(([key, value], index) => (
                      <tr key={index} style={{ backgroundColor: index % 2 === 0 ? 'rgba(255, 255, 255, 0.1)' : 'transparent' }}>
                        <td style={{ padding: '8px', color: 'white', fontWeight: 'bold' }}>
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </td>
                        <td style={{ padding: '8px', color: 'white' }}>
                          {value || 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PhishingDetector;
