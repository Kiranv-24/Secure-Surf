
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
  const [binaryColumns, setBinaryColumns] = useState([]);

  useEffect(() => {
    // Initialize binary rain columns
    const columns = [];
    const columnCount = Math.floor(window.innerWidth / 20);
    
    for (let i = 0; i < columnCount; i++) {
      columns.push({
        id: i,
        characters: [],
        speed: Math.random() * 3 + 1,
        opacity: Math.random() * 0.8 + 0.2
      });
    }
    setBinaryColumns(columns);

    // Update binary rain
    const interval = setInterval(() => {
      setBinaryColumns(prev => prev.map(column => ({
        ...column,
        characters: [
          Math.random() > 0.5 ? '1' : '0',
          ...column.characters.slice(0, 20)
        ]
      })));
    }, 150);

    return () => clearInterval(interval);
  }, []);

  // Simulate scanning progress when loading
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
    <div className="min-h-screen relative overflow-hidden bg-black">
      {/* Enhanced Binary Rain Background */}
      <div className="fixed inset-0 pointer-events-none">
        {/* Dark gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900/95 to-black z-10"></div>
        
        {/* Binary rain columns */}
        <div className="absolute inset-0 z-0">
          {binaryColumns.map((column) => (
            <div
              key={column.id}
              className="absolute top-0 font-mono text-green-400 select-none"
              style={{
                left: `${(column.id * 20)}px`,
                opacity: column.opacity,
                animation: `fall ${4 + column.speed}s linear infinite`,
                fontSize: '14px',
                lineHeight: '20px'
              }}
            >
              {column.characters.map((char, index) => (
                <div
                  key={index}
                  style={{
                    opacity: Math.max(0, 1 - (index * 0.1)),
                    color: index === 0 ? '#00ff00' : '#004400'
                  }}
                >
                  {char}
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Cyber grid overlay */}
        <div 
          className="absolute inset-0 opacity-10 z-5"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0, 255, 255, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 255, 255, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px',
            animation: 'gridMove 20s linear infinite'
          }}
        ></div>

        {/* Floating cyber particles */}
        <div className="absolute inset-0 z-5">
          {[...Array(25)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-cyan-400 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `float ${3 + Math.random() * 4}s ease-in-out infinite`,
                animationDelay: `${Math.random() * 3}s`,
                boxShadow: '0 0 10px rgba(0, 255, 255, 0.8)'
              }}
            ></div>
          ))}
        </div>

        {/* Scanning line effect */}
        <div className="absolute inset-0 opacity-20 z-5">
          <div 
            className="w-full h-0.5 bg-gradient-to-r from-transparent via-red-400 to-transparent"
            style={{
              animation: 'scanLine 4s linear infinite',
              position: 'absolute',
              top: '0',
              boxShadow: '0 0 20px rgba(255, 0, 0, 0.8)'
            }}
          ></div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-20 min-h-screen flex flex-col text-white">
        {/* Cyber Header */}
        <div className="p-8 border-b border-cyan-400/20 bg-black/90 backdrop-blur-xl">
          <div className="max-w-6xl mx-auto">
            {/* Terminal window simulation */}
            <div className="flex items-center space-x-4 mb-8">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse shadow-lg shadow-red-500/50"></div>
                <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse shadow-lg shadow-yellow-400/50"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"></div>
              </div>
              <div className="text-green-400 font-mono text-sm opacity-80">
                root@cybershield:~$ ./threat_detector --neural-engine --deep-scan
              </div>
            </div>
            
            <div className="text-center">
              {/* Enhanced Logo Section */}
              <div className="inline-flex items-center mb-8">
                <div className="relative">
                  <div className="w-20 h-20 bg-gradient-to-br from-red-500 via-purple-600 to-cyan-400 rounded-xl flex items-center justify-center mr-6 shadow-2xl shadow-cyan-500/30 relative overflow-hidden">
                    {/* Animated background effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-red-400/20 to-cyan-400/20 animate-pulse"></div>
                    <svg className="w-12 h-12 text-white z-10" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-4 1-1-4 .257-.257A6 6 0 0118 8zm-6-2a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    {/* Glowing border effect */}
                    <div className="absolute inset-0 rounded-xl border-2 border-cyan-400/50 animate-pulse"></div>
                  </div>
                </div>
                <div>
                  <h1 className="text-5xl md:text-7xl font-black mb-3">
                    <span className="bg-gradient-to-r from-red-400 via-purple-500 to-cyan-400 bg-clip-text text-transparent animate-pulse">
                      CYBER SHIELD
                    </span>
                  </h1>
                  <p className="text-cyan-400 font-mono tracking-widest text-lg opacity-90">
                    Neural Threat Detection System v3.0
                  </p>
                  <div className="flex justify-center mt-4">
                    <div className="px-4 py-2 bg-red-900/30 border border-red-500/50 rounded-full">
                      <span className="text-red-400 font-mono text-sm animate-pulse">● THREAT_ANALYSIS_ACTIVE</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Enhanced Status Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                <div className="bg-black/50 border border-green-500/30 rounded-lg p-4 backdrop-blur-sm">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></div>
                    <span className="text-green-400 font-mono text-sm">AI_ENGINE</span>
                  </div>
                  <div className="text-white font-bold">ONLINE</div>
                </div>
                
                <div className="bg-black/50 border border-blue-500/30 rounded-lg p-4 backdrop-blur-sm">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse shadow-lg shadow-blue-400/50"></div>
                    <span className="text-blue-400 font-mono text-sm">DEEP_SCAN</span>
                  </div>
                  <div className="text-white font-bold">READY</div>
                </div>
                
                <div className="bg-black/50 border border-purple-500/30 rounded-lg p-4 backdrop-blur-sm">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse shadow-lg shadow-purple-400/50"></div>
                    <span className="text-purple-400 font-mono text-sm">ML_MODEL</span>
                  </div>
                  <div className="text-white font-bold">ACTIVE</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Scanner Interface */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-4xl w-full">
            <div className="relative">
              {/* Glowing background effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-red-900/20 via-purple-900/20 to-cyan-900/20 rounded-3xl blur-xl"></div>
              
              <div className="relative bg-black/80 backdrop-blur-2xl border border-cyan-400/30 rounded-3xl p-8 shadow-2xl">
                
                {/* Enhanced Terminal Header */}
                <div className="flex items-center justify-between mb-8 pb-6 border-b border-gray-700/50">
                  <div className="flex items-center space-x-4">
                    <div className="text-cyan-400 text-xl animate-pulse">❯</div>
                    <span className="font-mono text-cyan-400 text-lg">neural_threat_analyzer.py</span>
                    <div className="px-3 py-1 bg-green-900/30 border border-green-500/50 rounded-full">
                      <span className="text-green-400 font-mono text-xs">RUNNING</span>
                    </div>
                  </div>
                  <div className="text-xs font-mono text-gray-400">
                    {new Date().toLocaleString()}
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-8">
                  {/* Enhanced URL Input */}
                  <div>
                    <div className="flex items-center space-x-3 mb-4">
                      <span className="text-cyan-400 font-mono">TARGET_URL:</span>
                      <span className="text-gray-500 font-mono text-sm animate-pulse">◉ SCANNING_MODE_ACTIVE</span>
                    </div>
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 via-purple-500/20 to-cyan-500/20 rounded-xl blur-sm opacity-0 group-focus-within:opacity-100 transition-all duration-500"></div>
                      <div className="relative flex items-center">
                        <div className="absolute left-4 flex items-center space-x-2">
                          <span className="text-green-400 font-mono">$</span>
                          <div className="w-2 h-4 bg-green-400 animate-pulse"></div>
                        </div>
                        <input
                          type="text"
                          value={url}
                          onChange={(e) => setUrl(e.target.value)}
                          placeholder="https://suspicious-domain.com"
                          className="w-full pl-12 pr-12 py-5 bg-black/70 border-2 border-gray-600 hover:border-cyan-400 focus:border-cyan-400 rounded-xl text-white placeholder-gray-500 focus:outline-none font-mono backdrop-blur-sm transition-all duration-300 text-lg"
                          disabled={loading}
                        />
                        {url && (
                          <button
                            type="button"
                            onClick={clearInput}
                            className="absolute right-4 text-gray-500 hover:text-red-400 transition-colors text-xl"
                          >
                            ✕
                          </button>
                        )}
                      </div>
                      <div className="absolute inset-0 rounded-xl border border-cyan-400/20 pointer-events-none"></div>
                    </div>
                  </div>

                  {/* Enhanced Scan Button */}
                  <button
                    type="submit"
                    disabled={loading || !url.trim()}
                    className="w-full py-5 bg-gradient-to-r from-red-600 via-purple-600 to-cyan-600 hover:from-red-500 hover:via-purple-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-white font-mono font-bold rounded-xl transition-all duration-300 border-2 border-cyan-500/50 hover:border-cyan-400/70 shadow-2xl relative overflow-hidden group text-lg"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-red-400/20 via-purple-400/20 to-cyan-400/20 animate-pulse opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    <div className="relative flex items-center justify-center space-x-3">
                      {loading ? (
                        <>
                          <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
                          <span>NEURAL_SCAN_ACTIVE...</span>
                        </>
                      ) : (
                        <>
                          <span>🔍</span>
                          <span>EXECUTE_DEEP_ANALYSIS</span>
                          <span>🔍</span>
                        </>
                      )}
                    </div>
                  </button>

                  {/* Enhanced Progress Bar */}
                  {loading && (
                    <div className="space-y-4">
                      <div className="flex justify-between text-sm font-mono text-cyan-400">
                        <span>NEURAL_NETWORK_ANALYSIS</span>
                        <span>{Math.round(scanProgress)}%</span>
                      </div>
                      <div className="h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-600">
                        <div 
                          className="h-full bg-gradient-to-r from-red-400 via-purple-500 to-cyan-400 transition-all duration-300 relative"
                          style={{ width: `${scanProgress}%` }}
                        >
                          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                        </div>
                      </div>
                      <div className="text-xs font-mono text-gray-400 animate-pulse text-center">
                        Analyzing threat patterns... Processing ML signatures... Running deep neural analysis...
                      </div>
                    </div>
                  )}
                </form>

                {/* Enhanced Error Display */}
                {error && (
                  <div className="mt-8 p-4 bg-red-900/30 border-2 border-red-500/50 rounded-xl flex items-center backdrop-blur-sm">
                    <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center mr-4 text-sm animate-pulse">⚠</div>
                    <span className="text-red-300 font-mono">{error}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-xl">
          <div className="bg-gradient-to-br from-gray-900/95 to-black/95 border-2 border-cyan-400/30 rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl backdrop-blur-2xl">
            
            {/* Enhanced Modal Header */}
            <div className={`p-8 border-b border-gray-700 relative overflow-hidden ${
              isPhishing ? 'bg-gradient-to-r from-red-900/40 to-orange-900/40' : 
              isLegitimate ? 'bg-gradient-to-r from-green-900/40 to-emerald-900/40' : 
              'bg-gradient-to-r from-gray-800/40 to-gray-700/40'
            }`}>
              <div className="flex items-center justify-between relative z-10">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-16 h-16 bg-red-500/20 border-2 border-red-500 rounded-xl flex items-center justify-center mr-6 animate-pulse">
                      <span className="text-3xl">⚠️</span>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-16 h-16 bg-green-500/20 border-2 border-green-500 rounded-xl flex items-center justify-center mr-6 animate-pulse">
                      <span className="text-3xl">✅</span>
                    </div>
                  ) : (
                    <div className="w-16 h-16 bg-yellow-500/20 border-2 border-yellow-500 rounded-xl flex items-center justify-center mr-6 animate-pulse">
                      <span className="text-3xl">🔍</span>
                    </div>
                  )}
                  <div>
                    <h2 className="text-3xl font-bold text-white font-mono">ANALYSIS_COMPLETE</h2>
                    <p className="text-gray-400 font-mono">Neural network scan results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-white transition-colors p-3 hover:bg-gray-700/50 rounded-xl text-xl"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Enhanced Modal Content */}
            <div className="p-8">
              {/* Main Result Display */}
              <div className={`p-8 rounded-2xl mb-8 border-2 relative overflow-hidden ${
                isPhishing ? 'bg-red-900/20 border-red-500/50' : 
                isLegitimate ? 'bg-green-900/20 border-green-500/50' : 
                'bg-yellow-900/20 border-yellow-500/50'
              }`}>
                <div className="font-mono text-center relative z-10">
                  <div className="text-sm text-gray-400 mb-3">THREAT_STATUS:</div>
                  <div className={`text-2xl font-bold ${
                    isPhishing ? 'text-red-300' : 
                    isLegitimate ? 'text-green-300' : 
                    'text-yellow-300'
                  }`}>
                    {result.result_text || 'ANALYSIS_COMPLETE'}
                  </div>
                </div>
              </div>

              {/* Detailed Analysis */}
              {result.additional_info && Object.keys(result.additional_info).length > 0 && (
                <div>
                  <h3 className="text-2xl font-bold text-white mb-6 font-mono flex items-center">
                    <span className="text-cyan-400 mr-3">📊</span>
                    DETAILED_REPORT
                  </h3>
                  
                  <div className="bg-black/40 rounded-2xl border border-gray-700/50 p-6">
                    <div className="space-y-4">
                      {Object.entries(result.additional_info).map(([key, value], index) => (
                        <div key={index} className="flex justify-between items-center py-3 border-b border-gray-800 last:border-b-0">
                          <div className="text-cyan-400 font-mono uppercase font-bold">
                            {key.replace(/_/g, '_')}:
                          </div>
                          <div className="text-white font-mono bg-gray-800/50 px-3 py-1 rounded-lg">
                            {value || 'NULL'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Enhanced Modal Footer */}
            <div className="p-8 border-t border-gray-700 bg-black/30">
              <button
                onClick={closeModal}
                className="w-full py-4 bg-gradient-to-r from-gray-700 via-gray-800 to-gray-700 hover:from-gray-600 hover:via-gray-700 hover:to-gray-600 text-white font-mono rounded-xl transition-all duration-300 border border-gray-600 text-lg font-bold"
              >
                CLOSE_TERMINAL
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Custom Animations */}
      <style jsx>{`
        @keyframes fall {
          0% { transform: translateY(-100vh); }
          100% { transform: translateY(100vh); }
        }
        
        @keyframes gridMove {
          0% { transform: translate(0, 0); }
          100% { transform: translate(40px, 40px); }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-30px) rotate(180deg); }
        }
        
        @keyframes scanLine {
          0% { top: 0%; opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export default PhishingDetector;
