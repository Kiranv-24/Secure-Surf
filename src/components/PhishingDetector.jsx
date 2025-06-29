
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);

  // Binary rain effect
  const [binaryRain, setBinaryRain] = useState([]);

  useEffect(() => {
    // Initialize binary rain
    const columns = Math.floor(window.innerWidth / 15);
    const rainArray = [];
    
    for (let i = 0; i < columns; i++) {
      rainArray.push({
        id: i,
        text: '',
        y: Math.random() * -100,
        speed: Math.random() * 3 + 1
      });
    }
    setBinaryRain(rainArray);

    // Animate binary rain
    const interval = setInterval(() => {
      setBinaryRain(prev => prev.map(drop => ({
        ...drop,
        text: Math.random() > 0.5 ? '1' : '0',
        y: drop.y > window.innerHeight ? -20 : drop.y + drop.speed
      })));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  // Simulate scanning progress
  useEffect(() => {
    if (loading) {
      setScanProgress(0);
      const interval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return 95;
          }
          return prev + Math.random() * 15;
        });
      }, 200);
      return () => clearInterval(interval);
    }
  }, [loading]);

  const checkPhishing = async () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setShowModal(false);

    try {
      const response = await axios.post('http://127.0.0.1:2002/check_phishing', {
        url: url.trim()
      });
      
      setTimeout(() => {
        setScanProgress(100);
        setResult(response.data);
        setShowModal(true);
      }, 1000);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to analyze URL. Please check your connection and try again.');
    } finally {
      setTimeout(() => {
        setLoading(false);
        setScanProgress(0);
      }, 1200);
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
    setShowModal(false);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const isPhishing = result?.result_text?.toLowerCase().includes('phishing');
  const isLegitimate = result?.result_text?.toLowerCase().includes('legitimate');

  return (
    <div className="min-h-screen bg-black text-green-400 relative overflow-hidden font-mono">
      {/* Binary Rain Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {binaryRain.map((drop) => (
          <div
            key={drop.id}
            className="absolute text-green-400 text-sm opacity-60 animate-pulse"
            style={{
              left: `${drop.id * 15}px`,
              top: `${drop.y}px`,
              textShadow: '0 0 10px #00ff00'
            }}
          >
            {drop.text}
          </div>
        ))}
      </div>

      {/* Dark overlay */}
      <div className="fixed inset-0 bg-black/70 z-10"></div>

      {/* Main Content */}
      <div className="relative z-20 min-h-screen flex flex-col items-center justify-center p-4">
        
        {/* Cyber Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-green-400 rounded-lg flex items-center justify-center mr-4 shadow-lg shadow-green-400/50 border border-green-400/50">
              <span className="text-2xl">🛡️</span>
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 to-red-400 bg-clip-text text-transparent">
                CYBER SHIELD
              </h1>
              <p className="text-green-400/80 text-sm">Neural Threat Detection</p>
            </div>
          </div>
          
          <div className="flex justify-center space-x-4 text-xs">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-1"></div>
              <span>AI_ENGINE_ONLINE</span>
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1"></div>
              <span>DEEP_SCAN_READY</span>
            </div>
          </div>
        </div>

        {/* Compact Scanner Interface */}
        <div className="w-full max-w-2xl">
          <div className="bg-black/80 border border-green-400/30 rounded-lg p-6 backdrop-blur-sm shadow-2xl shadow-green-400/20">
            
            {/* Terminal Header */}
            <div className="flex items-center mb-4 pb-3 border-b border-green-400/20">
              <div className="flex space-x-2 mr-4">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <span className="text-green-400 text-sm">root@cybershield:~$ threat_analyzer.py</span>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* URL Input */}
              <div>
                <div className="flex items-center mb-2">
                  <span className="text-green-400 mr-2">TARGET_URL:</span>
                  <div className="w-2 h-4 bg-green-400 animate-pulse"></div>
                </div>
                <div className="relative">
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://suspicious-domain.com"
                    className="w-full px-4 py-3 bg-black/70 border border-green-400/50 hover:border-green-400 focus:border-green-400 rounded text-green-400 placeholder-green-400/50 focus:outline-none font-mono backdrop-blur-sm transition-all"
                    disabled={loading}
                  />
                  {url && (
                    <button
                      type="button"
                      onClick={clearInput}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-green-400/70 hover:text-red-400 transition-colors"
                    >
                      ✕
                    </button>
                  )}
                </div>
              </div>

              {/* Scan Button */}
              <button
                type="submit"
                disabled={loading || !url.trim()}
                className="w-full py-3 bg-gradient-to-r from-green-600 to-red-600 hover:from-green-500 hover:to-red-500 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-black font-bold rounded transition-all duration-300 border border-green-400/50 hover:border-green-400 shadow-lg hover:shadow-green-400/30 relative overflow-hidden"
              >
                <div className="relative flex items-center justify-center space-x-2">
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-2 border-black border-t-transparent"></div>
                      <span>SCANNING...</span>
                    </>
                  ) : (
                    <>
                      <span>🔍</span>
                      <span>EXECUTE_SCAN</span>
                    </>
                  )}
                </div>
              </button>

              {/* Progress Bar */}
              {loading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>NEURAL_ANALYSIS</span>
                    <span>{Math.round(scanProgress)}%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded overflow-hidden border border-green-400/30">
                    <div 
                      className="h-full bg-gradient-to-r from-green-400 to-red-400 transition-all duration-300"
                      style={{ width: `${scanProgress}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-green-400/70 animate-pulse text-center">
                    Analyzing threat patterns...
                  </div>
                </div>
              )}
            </form>

            {/* Error Display */}
            {error && (
              <div className="mt-4 p-3 bg-red-900/30 border border-red-500/50 rounded flex items-center">
                <span className="text-red-400 mr-2">⚠</span>
                <span className="text-red-300 text-sm">{error}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-xl">
          <div className="bg-black/90 border border-green-400/50 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-2xl shadow-green-400/30">
            
            {/* Modal Header */}
            <div className={`p-6 border-b border-green-400/30 ${
              isPhishing ? 'bg-red-900/20' : 
              isLegitimate ? 'bg-green-900/20' : 
              'bg-gray-800/20'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-12 h-12 bg-red-500/20 border border-red-500 rounded flex items-center justify-center mr-4">
                      <span className="text-2xl">⚠️</span>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-12 h-12 bg-green-500/20 border border-green-500 rounded flex items-center justify-center mr-4">
                      <span className="text-2xl">✅</span>
                    </div>
                  ) : (
                    <div className="w-12 h-12 bg-yellow-500/20 border border-yellow-500 rounded flex items-center justify-center mr-4">
                      <span className="text-2xl">🔍</span>
                    </div>
                  )}
                  <div>
                    <h2 className="text-xl font-bold text-green-400">SCAN_COMPLETE</h2>
                    <p className="text-green-400/70 text-sm">Analysis results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-green-400/70 hover:text-red-400 transition-colors p-2 text-xl"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {/* Main Result */}
              <div className={`p-4 rounded border mb-4 ${
                isPhishing ? 'bg-red-900/20 border-red-500/50' : 
                isLegitimate ? 'bg-green-900/20 border-green-500/50' : 
                'bg-yellow-900/20 border-yellow-500/50'
              }`}>
                <div className="text-center">
                  <div className="text-xs text-green-400/70 mb-1">THREAT_STATUS:</div>
                  <div className={`text-lg font-bold ${
                    isPhishing ? 'text-red-400' : 
                    isLegitimate ? 'text-green-400' : 
                    'text-yellow-400'
                  }`}>
                    {result.result_text || 'ANALYSIS_COMPLETE'}
                  </div>
                </div>
              </div>

              {/* Additional Info */}
              {result.additional_info && Object.keys(result.additional_info).length > 0 && (
                <div>
                  <h3 className="text-green-400 font-bold mb-3 flex items-center">
                    <span className="mr-2">📊</span>
                    DETAILED_REPORT
                  </h3>
                  
                  <div className="bg-black/40 rounded border border-green-400/30 p-4">
                    <div className="space-y-2">
                      {Object.entries(result.additional_info).map(([key, value], index) => (
                        <div key={index} className="flex justify-between items-center py-2 border-b border-green-400/20 last:border-b-0">
                          <div className="text-green-400 text-sm font-bold uppercase">
                            {key.replace(/_/g, '_')}:
                          </div>
                          <div className="text-green-400/80 bg-black/50 px-2 py-1 rounded text-sm">
                            {value || 'NULL'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-green-400/30">
              <button
                onClick={closeModal}
                className="w-full py-3 bg-green-600 hover:bg-green-500 text-black font-bold rounded transition-all duration-300 border border-green-400/50"
              >
                CLOSE_TERMINAL
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Custom Styles */}
      <style jsx>{`
        @keyframes matrix {
          0% { transform: translateY(-100vh); }
          100% { transform: translateY(100vh); }
        }
      `}</style>
    </div>
  );
};

export default PhishingDetector;
