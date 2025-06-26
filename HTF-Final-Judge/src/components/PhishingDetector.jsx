
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
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        {/* Matrix-style falling code */}
        <div className="absolute inset-0 opacity-5">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-pulse"
              style={{
                left: `${i * 5}%`,
                animationDelay: `${i * 0.5}s`,
                animationDuration: '3s'
              }}
            >
              {[...Array(50)].map((_, j) => (
                <div key={j} className="text-green-400 text-xs font-mono opacity-30">
                  {Math.random() > 0.5 ? '1' : '0'}
                </div>
              ))}
            </div>
          ))}
        </div>
        
        {/* Cyber grid overlay */}
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `
            radial-gradient(circle at 1px 1px, rgba(0,255,255,0.3) 1px, transparent 0),
            linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '100px 100px, 50px 50px, 50px 50px'
        }}></div>
      </div>

      {/* Header Section */}
      <div className="relative z-10">
        <div className="text-center pt-8 pb-6">
          {/* Logo */}
          <div className="inline-flex items-center justify-center mb-6">
            <div className="relative">
              <div className="w-20 h-20 bg-gradient-to-r from-red-500 via-orange-500 to-yellow-500 rounded-full flex items-center justify-center shadow-lg animate-pulse">
                <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-4 1-1-4 .257-.257A6 6 0 0118 8zm-6-2a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
              </div>
              {/* Scanning rings */}
              <div className="absolute inset-0 rounded-full border-2 border-cyan-400 animate-ping"></div>
              <div className="absolute inset-0 rounded-full border border-green-400 animate-pulse" style={{animationDelay: '0.5s'}}></div>
            </div>
          </div>

          {/* Title */}
          <h1 className="text-5xl md:text-7xl font-black mb-4">
            <span className="bg-gradient-to-r from-red-400 via-yellow-400 to-orange-500 bg-clip-text text-transparent">
              CYBER
            </span>
            <span className="text-white mx-2">·</span>
            <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
              SHIELD
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-4 font-light">Advanced Threat Detection System</p>
          
          {/* Status indicators */}
          <div className="flex items-center justify-center space-x-8 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-400 font-semibold">AI ONLINE</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
              <span className="text-blue-400 font-semibold">SCANNING ACTIVE</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
              <span className="text-red-400 font-semibold">THREATS BLOCKED</span>
            </div>
          </div>
        </div>

        {/* Main Scanner Interface */}
        <div className="max-w-6xl mx-auto px-6">
          <div className="relative">
            {/* Main scanning panel */}
            <div className="bg-gradient-to-br from-gray-900/90 via-gray-800/90 to-gray-900/90 backdrop-blur-xl border border-gray-600/50 rounded-3xl p-8 shadow-2xl">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-white mb-2">THREAT ANALYSIS CONSOLE</h2>
                <div className="w-20 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full mx-auto"></div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="relative">
                  <label className="block text-sm font-semibold text-cyan-400 mb-3 uppercase tracking-wider">
                    TARGET URL INPUT
                  </label>
                  <div className="relative group">
                    <input
                      type="text"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://suspicious-website.com"
                      className="w-full px-6 py-4 bg-black/50 border-2 border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500 focus:shadow-lg focus:shadow-cyan-500/25 transition-all duration-300 font-mono text-lg"
                      disabled={loading}
                    />
                    {url && (
                      <button
                        type="button"
                        onClick={clearInput}
                        className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-red-400 transition-colors"
                      >
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                {/* Scan Button */}
                <button
                  type="submit"
                  disabled={loading || !url.trim()}
                  className="w-full py-4 bg-gradient-to-r from-red-600 via-orange-600 to-yellow-600 hover:from-red-700 hover:via-orange-700 hover:to-yellow-700 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-white font-bold text-lg rounded-xl transition-all duration-300 transform hover:scale-[1.02] disabled:scale-100 shadow-xl uppercase tracking-wider"
                >
                  {loading ? (
                    <div className="flex items-center justify-center space-x-3">
                      <div className="animate-spin rounded-full h-6 w-6 border-3 border-white border-t-transparent"></div>
                      <span>SCANNING IN PROGRESS...</span>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center space-x-3">
                      <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span>INITIATE DEEP SCAN</span>
                    </div>
                  )}
                </button>

                {/* Progress Bar */}
                {loading && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm text-cyan-400">
                      <span>Scanning Progress</span>
                      <span>{Math.round(scanProgress)}%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                        style={{ width: `${scanProgress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </form>

              {/* Error Display */}
              {error && (
                <div className="mt-6 p-4 bg-red-900/50 border border-red-500 rounded-xl flex items-center">
                  <svg className="w-6 h-6 text-red-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span className="text-red-300 font-semibold">{error}</span>
                </div>
              )}
            </div>
          </div>

          {/* Threat Intelligence Dashboard */}
          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Legitimate Sites */}
            <div className="bg-gradient-to-br from-green-900/30 via-green-800/30 to-emerald-900/30 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-green-400 mb-4 flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></div>
                VERIFIED SAFE DOMAINS
              </h3>
              <div className="space-y-2">
                {[
                  'https://google.com',
                  'https://github.com',
                  'https://stackoverflow.com',
                  'https://microsoft.com',
                  'https://amazon.com'
                ].map((site, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-green-900/20 rounded-lg">
                    <span className="text-green-300 font-mono text-sm">{site}</span>
                    <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded-full text-xs font-semibold">
                      SAFE
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Malicious Sites */}
            <div className="bg-gradient-to-br from-red-900/30 via-red-800/30 to-orange-900/30 backdrop-blur-xl border border-red-500/30 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-red-400 mb-4 flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-3 animate-pulse"></div>
                DETECTED THREATS
              </h3>
              <div className="space-y-2">
                {[
                  'http://malicious-site.ru',
                  'https://fake-bank.scam',
                  'http://phishing-attempt.org',
                  'https://virus-download.net',
                  'http://stolen-credentials.com'
                ].map((threat, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-red-900/20 rounded-lg">
                    <span className="text-red-300 font-mono text-sm">{threat}</span>
                    <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded-full text-xs font-semibold">
                      BLOCKED
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 border-2 border-gray-600/50 rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            {/* Modal Header */}
            <div className={`p-8 border-b border-gray-700 ${
              isPhishing ? 'bg-gradient-to-r from-red-900/50 to-orange-900/50' : 
              isLegitimate ? 'bg-gradient-to-r from-green-900/50 to-emerald-900/50' : 
              'bg-gradient-to-r from-gray-800/50 to-gray-700/50'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-orange-500 rounded-full flex items-center justify-center mr-6 animate-pulse">
                      <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center mr-6 animate-pulse">
                      <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                  ) : (
                    <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full flex items-center justify-center mr-6 animate-pulse">
                      <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                    </div>
                  )}
                  <div>
                    <h2 className="text-3xl font-bold text-white mb-2">SCAN COMPLETE</h2>
                    <p className="text-gray-300 text-lg">Threat analysis results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-white transition-colors p-2"
                >
                  <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-8">
              {/* Main Result */}
              <div className={`p-6 rounded-2xl mb-8 text-center ${
                isPhishing ? 'bg-gradient-to-r from-red-900/50 to-orange-900/50 border border-red-500/50' : 
                isLegitimate ? 'bg-gradient-to-r from-green-900/50 to-emerald-900/50 border border-green-500/50' : 
                'bg-gradient-to-r from-yellow-900/50 to-orange-900/50 border border-yellow-500/50'
              }`}>
                <p className={`text-2xl font-bold ${
                  isPhishing ? 'text-red-300' : 
                  isLegitimate ? 'text-green-300' : 
                  'text-yellow-300'
                }`}>
                  {result.result_text || 'Analysis complete'}
                </p>
              </div>

              {/* Detailed Analysis */}
              {result.additional_info && Object.keys(result.additional_info).length > 0 && (
                <div>
                  <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                    <svg className="w-6 h-6 text-cyan-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" clipRule="evenodd" />
                    </svg>
                    DETAILED ANALYSIS
                  </h3>
                  
                  <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl overflow-hidden border border-gray-600/50">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-6">
                      {Object.entries(result.additional_info).map(([key, value], index) => (
                        <div key={index} className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                          <div className="text-sm font-semibold text-cyan-400 mb-2 uppercase tracking-wider">
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                          <div className="text-white font-mono text-lg">
                            {value || 'N/A'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-8 border-t border-gray-700 bg-gray-800/30">
              <button
                onClick={closeModal}
                className="w-full py-4 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white font-bold text-lg rounded-xl transition-all duration-300 uppercase tracking-wider"
              >
                CLOSE ANALYSIS
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhishingDetector;
