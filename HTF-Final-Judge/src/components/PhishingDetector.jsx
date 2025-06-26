import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);

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
      {/* Matrix Digital Rain Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black"></div>
        
        {/* Animated Matrix Rain */}
        <div className="absolute inset-0 opacity-20">
          {[...Array(100)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-pulse"
              style={{
                left: `${(i * 10) % 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${3 + Math.random() * 4}s`
              }}
            >
              <div className="text-green-400 font-mono text-sm leading-none">
                {['0', '1', '0', '1', '0', '1'].map((digit, idx) => (
                  <div key={idx} className="opacity-60" style={{ animationDelay: `${idx * 0.1}s` }}>
                    {digit}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Cyber Grid */}
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0, 255, 255, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 255, 255, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px',
            animation: 'moveGrid 20s linear infinite'
          }}
        ></div>

        {/* Floating Cyber Particles */}
        <div className="absolute inset-0">
          {[...Array(30)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-cyan-400 rounded-full shadow-lg shadow-cyan-400/50"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `float ${3 + Math.random() * 4}s ease-in-out infinite`,
                animationDelay: `${Math.random() * 3}s`
              }}
            ></div>
          ))}
        </div>

        {/* Holographic Scan Lines */}
        <div className="absolute inset-0 opacity-30">
          <div 
            className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-lg shadow-cyan-400/50"
            style={{
              animation: 'scanLine 3s linear infinite',
              position: 'absolute',
              top: '0'
            }}
          ></div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex flex-col text-white">
        {/* Terminal Header */}
        <div className="p-8 border-b border-cyan-400/30 bg-black/80 backdrop-blur-xl">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center space-x-4 mb-6">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full shadow-lg shadow-red-500/50"></div>
                <div className="w-3 h-3 bg-yellow-400 rounded-full shadow-lg shadow-yellow-400/50"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full shadow-lg shadow-green-500/50"></div>
              </div>
              <div className="text-gray-400 font-mono text-sm">
                root@cybershield:~$ ./phishing_detector --deep-learning
              </div>
            </div>
            
            <div className="text-center">
              <div className="inline-flex items-center mb-6">
                <div className="w-16 h-16 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg flex items-center justify-center mr-4 shadow-2xl shadow-cyan-500/50">
                  <svg className="w-10 h-10 text-black" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-4 1-1-4 .257-.257A6 6 0 0118 8zm-6-2a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-4xl md:text-6xl font-black mb-2">
                    <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                      CYBER SHIELD
                    </span>
                  </h1>
                  <p className="text-cyan-400 font-mono tracking-wider">
                    Deep Learning Threat Detection System v2.1.4
                  </p>
                </div>
              </div>
              
              {/* Status Indicators */}
              <div className="flex justify-center space-x-8 text-sm font-mono">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></div>
                  <span className="text-green-400">NEURAL_NET_ONLINE</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse shadow-lg shadow-blue-400/50"></div>
                  <span className="text-blue-400">DEEP_SCAN_READY</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse shadow-lg shadow-purple-400/50"></div>
                  <span className="text-purple-400">ML_MODEL_ACTIVE</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Scanner Interface */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-4xl w-full">
            <div className="bg-gradient-to-br from-gray-900/90 to-black/90 backdrop-blur-2xl border border-cyan-400/30 rounded-2xl p-8 shadow-2xl shadow-cyan-500/20">
              
              {/* Terminal Window Header */}
              <div className="flex items-center justify-between mb-8 pb-4 border-b border-gray-700">
                <div className="flex items-center space-x-3">
                  <div className="text-cyan-400">❯</div>
                  <span className="font-mono text-cyan-400">threat_analyzer.py</span>
                </div>
                <div className="text-xs font-mono text-gray-500">
                  {new Date().toLocaleString()}
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                {/* URL Input Terminal Style */}
                <div>
                  <div className="flex items-center space-x-2 mb-3">
                    <span className="text-cyan-400 font-mono text-sm">INPUT:</span>
                    <span className="text-gray-500 font-mono text-sm">target_url</span>
                  </div>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg blur-sm opacity-0 group-focus-within:opacity-100 transition-all duration-300"></div>
                    <div className="relative flex items-center">
                      <span className="absolute left-4 text-green-400 font-mono">$</span>
                      <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://suspicious-domain.com"
                        className="w-full pl-8 pr-12 py-4 bg-black/70 border border-gray-600 hover:border-cyan-400 focus:border-cyan-400 rounded-lg text-white placeholder-gray-500 focus:outline-none font-mono backdrop-blur-sm transition-all duration-300"
                        disabled={loading}
                      />
                      {url && (
                        <button
                          type="button"
                          onClick={clearInput}
                          className="absolute right-4 text-gray-500 hover:text-red-400 transition-colors"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Scan Button */}
                <button
                  type="submit"
                  disabled={loading || !url.trim()}
                  className="w-full py-4 bg-gradient-to-r from-red-600/80 to-orange-600/80 hover:from-red-500 hover:to-orange-500 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-white font-mono font-bold rounded-lg transition-all duration-300 border border-red-500/50 hover:border-red-400/70 shadow-lg shadow-red-500/25 relative overflow-hidden group"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-red-400/10 to-orange-400/10 animate-pulse opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  <div className="relative">
                    {loading ? (
                      <div className="flex items-center justify-center space-x-3">
                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                        <span>DEEP_SCAN_ACTIVE...</span>
                      </div>
                    ) : (
                      <span> EXECUTE_DEEP_ANALYSIS</span>
                    )}
                  </div>
                </button>

                {/* Progress Bar */}
                {loading && (
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm font-mono text-cyan-400">
                      <span>NEURAL_NETWORK_ANALYSIS</span>
                      <span>{Math.round(scanProgress)}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden border border-gray-600">
                      <div 
                        className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-300 shadow-lg shadow-cyan-400/50"
                        style={{ width: `${scanProgress}%` }}
                      ></div>
                    </div>
                    <div className="text-xs font-mono text-gray-500 animate-pulse">
                      Analyzing behavioral patterns... Checking ML signatures... Processing neural network...
                    </div>
                  </div>
                )}
              </form>

              {/* Error Display */}
              {error && (
                <div className="mt-6 p-4 bg-red-900/30 border border-red-500/50 rounded-lg flex items-center backdrop-blur-sm">
                  <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center mr-3 text-xs">!</div>
                  <span className="text-red-300 font-mono text-sm">{error}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-xl">
          <div className="bg-gradient-to-br from-gray-900/95 to-black/95 border border-cyan-400/30 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl backdrop-blur-2xl">
            
            {/* Modal Header */}
            <div className={`p-6 border-b border-gray-700 ${
              isPhishing ? 'bg-gradient-to-r from-red-900/40 to-orange-900/40' : 
              isLegitimate ? 'bg-gradient-to-r from-green-900/40 to-emerald-900/40' : 
              'bg-gradient-to-r from-gray-800/40 to-gray-700/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-12 h-12 bg-red-500/20 border border-red-500 rounded-lg flex items-center justify-center mr-4">
                      <span className="text-2xl">⚠️</span>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-12 h-12 bg-green-500/20 border border-green-500 rounded-lg flex items-center justify-center mr-4">
                      <span className="text-2xl">✓</span>
                    </div>
                  ) : (
                    <div className="w-12 h-12 bg-yellow-500/20 border border-yellow-500 rounded-lg flex items-center justify-center mr-4">
                      <span className="text-2xl">🔍</span>
                    </div>
                  )}
                  <div>
                    <h2 className="text-2xl font-bold text-white font-mono">ANALYSIS_COMPLETE</h2>
                    <p className="text-gray-400 font-mono text-sm">Deep learning scan results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-700/50 rounded-lg"
                >
                  <span className="text-xl">✕</span>
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {/* Main Result */}
              <div className={`p-6 rounded-xl mb-6 border ${
                isPhishing ? 'bg-red-900/20 border-red-500/50' : 
                isLegitimate ? 'bg-green-900/20 border-green-500/50' : 
                'bg-yellow-900/20 border-yellow-500/50'
              }`}>
                <div className="font-mono text-center">
                  <div className="text-sm text-gray-400 mb-2">THREAT_STATUS:</div>
                  <div className={`text-xl font-bold ${
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
                  <h3 className="text-xl font-bold text-white mb-4 font-mono flex items-center">
                    <span className="text-cyan-400 mr-2"></span>
                    DETAILED_REPORT
                  </h3>
                  
                  <div className="bg-black/40 rounded-xl border border-gray-700/50 p-4">
                    <div className="space-y-3">
                      {Object.entries(result.additional_info).map(([key, value], index) => (
                        <div key={index} className="flex justify-between items-center py-2 border-b border-gray-800 last:border-b-0">
                          <div className="text-sm text-cyan-400 font-mono uppercase">
                            {key.replace(/_/g, '_')}:
                          </div>
                          <div className="text-white font-mono text-sm">
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
            <div className="p-6 border-t border-gray-700 bg-black/30">
              <button
                onClick={closeModal}
                className="w-full py-3 bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-600 hover:to-gray-700 text-white font-mono rounded-lg transition-all duration-300 border border-gray-600"
              >
                CLOSE_TERMINAL
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Custom Animations */}
      <style jsx>{`
        @keyframes moveGrid {
          0% { transform: translate(0, 0); }
          100% { transform: translate(50px, 50px); }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
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