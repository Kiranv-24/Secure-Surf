
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
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated Cyber Background */}
      <div className="absolute inset-0">
        {/* Base gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900 via-blue-900 to-black"></div>
        
        {/* Animated grid overlay */}
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: `
            linear-gradient(rgba(0,255,255,0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,255,0.3) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'moveGrid 20s linear infinite'
        }}></div>

        {/* Floating particles */}
        <div className="absolute inset-0">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-pulse"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 3}s`
              }}
            ></div>
          ))}
        </div>

        {/* Animated waves */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -inset-10 opacity-30">
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20 transform rotate-12 animate-pulse"></div>
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-pink-600/20 via-red-600/20 to-orange-600/20 transform -rotate-12 animate-pulse" style={{animationDelay: '1s'}}></div>
          </div>
        </div>

        {/* Circuit pattern overlay */}
        <div className="absolute inset-0 opacity-10">
          <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="circuit" patternUnits="userSpaceOnUse" width="400" height="400">
                <g fill="none" stroke="currentColor" strokeWidth="1">
                  <path d="m 0,200 h 400 M 200,0 v 400" className="text-cyan-400"/>
                  <circle cx="200" cy="200" r="50" className="text-green-400"/>
                  <circle cx="100" cy="100" r="20" className="text-red-400"/>
                  <circle cx="300" cy="300" r="30" className="text-yellow-400"/>
                  <path d="M 50,50 L 150,150 M 250,50 L 350,150" className="text-purple-400"/>
                </g>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#circuit)"/>
          </svg>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 text-white min-h-screen flex flex-col">
        {/* Header Section with Enhanced Design */}
        <div className="text-center pt-12 pb-8">
          {/* Futuristic Logo */}
          <div className="inline-flex items-center justify-center mb-8">
            <div className="relative">
              {/* Main logo circle */}
              <div className="w-24 h-24 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-2xl shadow-cyan-500/50 animate-pulse">
                <svg className="w-12 h-12 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-4 1-1-4 .257-.257A6 6 0 0118 8zm-6-2a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
              </div>
              {/* Animated rings */}
              <div className="absolute inset-0 rounded-full border-2 border-cyan-400/50 animate-ping"></div>
              <div className="absolute -inset-2 rounded-full border border-purple-400/30 animate-pulse" style={{animationDelay: '0.5s'}}></div>
              <div className="absolute -inset-4 rounded-full border border-blue-400/20 animate-pulse" style={{animationDelay: '1s'}}></div>
            </div>
          </div>

          {/* Enhanced Title */}
          <h1 className="text-6xl md:text-8xl font-black mb-6 relative">
            <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent animate-pulse">
              CYBER
            </span>
            <span className="text-white mx-4 animate-pulse" style={{animationDelay: '0.5s'}}>⚡</span>
            <span className="bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 bg-clip-text text-transparent animate-pulse" style={{animationDelay: '1s'}}>
              SHIELD
            </span>
          </h1>
          
          <div className="relative mb-8">
            <p className="text-2xl text-gray-300 mb-4 font-light tracking-wider">
              Advanced Threat Detection System
            </p>
            <div className="w-32 h-1 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-full mx-auto animate-pulse"></div>
          </div>
          
          {/* Enhanced Status indicators */}
          <div className="flex items-center justify-center space-x-12 text-lg font-semibold">
            <div className="flex items-center space-x-3 bg-green-900/30 backdrop-blur-sm px-4 py-2 rounded-full border border-green-500/30">
              <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"></div>
              <span className="text-green-400">AI ONLINE</span>
            </div>
            <div className="flex items-center space-x-3 bg-blue-900/30 backdrop-blur-sm px-4 py-2 rounded-full border border-blue-500/30">
              <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse shadow-lg shadow-blue-500/50" style={{animationDelay: '0.5s'}}></div>
              <span className="text-blue-400">SCANNING ACTIVE</span>
            </div>
            <div className="flex items-center space-x-3 bg-red-900/30 backdrop-blur-sm px-4 py-2 rounded-full border border-red-500/30">
              <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse shadow-lg shadow-red-500/50" style={{animationDelay: '1s'}}></div>
              <span className="text-red-400">THREATS BLOCKED</span>
            </div>
          </div>
        </div>

        {/* Main Scanner Interface - Enhanced */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="max-w-4xl w-full">
            <div className="relative">
              {/* Glowing background effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 rounded-3xl blur-xl"></div>
              
              {/* Main scanning panel */}
              <div className="relative bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-2xl border-2 border-gray-600/50 rounded-3xl p-10 shadow-2xl">
                <div className="text-center mb-10">
                  <h2 className="text-3xl font-bold text-white mb-4 tracking-wider">THREAT ANALYSIS CONSOLE</h2>
                  <div className="flex justify-center space-x-2">
                    <div className="w-8 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"></div>
                    <div className="w-8 h-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></div>
                    <div className="w-8 h-1 bg-gradient-to-r from-pink-500 to-red-500 rounded-full"></div>
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-8">
                  <div className="relative">
                    <label className="block text-lg font-bold text-cyan-400 mb-4 uppercase tracking-widest">
                      🎯 TARGET URL INPUT
                    </label>
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 rounded-xl blur-sm group-focus-within:blur-none transition-all duration-300"></div>
                      <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://suspicious-website.com"
                        className="relative w-full px-8 py-6 bg-black/70 border-2 border-gray-600 hover:border-cyan-500 focus:border-cyan-500 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:shadow-xl focus:shadow-cyan-500/25 transition-all duration-300 font-mono text-xl backdrop-blur-sm"
                        disabled={loading}
                      />
                      {url && (
                        <button
                          type="button"
                          onClick={clearInput}
                          className="absolute right-6 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-red-400 transition-colors text-2xl"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Enhanced Scan Button */}
                  <button
                    type="submit"
                    disabled={loading || !url.trim()}
                    className="relative w-full py-6 bg-gradient-to-r from-red-600 via-orange-600 to-yellow-600 hover:from-red-500 hover:via-orange-500 hover:to-yellow-500 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-white font-bold text-xl rounded-xl transition-all duration-300 transform hover:scale-[1.02] disabled:scale-100 shadow-2xl shadow-orange-500/25 uppercase tracking-wider overflow-hidden"
                  >
                    {/* Button glow effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-red-400/20 via-orange-400/20 to-yellow-400/20 animate-pulse"></div>
                    
                    <div className="relative">
                      {loading ? (
                        <div className="flex items-center justify-center space-x-4">
                          <div className="animate-spin rounded-full h-8 w-8 border-4 border-white border-t-transparent"></div>
                          <span>⚡ DEEP SCAN IN PROGRESS...</span>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center space-x-4">
                          <span>🛡️</span>
                          <span>INITIATE QUANTUM SCAN</span>
                          <span>🔍</span>
                        </div>
                      )}
                    </div>
                  </button>

                  {/* Enhanced Progress Bar */}
                  {loading && (
                    <div className="space-y-4">
                      <div className="flex justify-between text-lg text-cyan-400 font-semibold">
                        <span>🔄 Quantum Analysis Progress</span>
                        <span>{Math.round(scanProgress)}%</span>
                      </div>
                      <div className="h-4 bg-gray-800 rounded-full overflow-hidden border border-gray-600">
                        <div 
                          className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 transition-all duration-300 shadow-lg shadow-cyan-500/50"
                          style={{ width: `${scanProgress}%` }}
                        ></div>
                      </div>
                      <div className="text-center text-sm text-gray-400 animate-pulse">
                        Analyzing threat patterns... Checking malicious signatures... Validating domain integrity...
                      </div>
                    </div>
                  )}
                </form>

                {/* Enhanced Error Display */}
                {error && (
                  <div className="mt-8 p-6 bg-gradient-to-r from-red-900/60 to-orange-900/60 border-2 border-red-500/50 rounded-xl flex items-center backdrop-blur-sm">
                    <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center mr-4 animate-pulse">
                      ⚠️
                    </div>
                    <span className="text-red-300 font-semibold text-lg">{error}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Futuristic Stats Bar */}
        <div className="py-8">
          <div className="max-w-6xl mx-auto px-6">
            <div className="bg-gradient-to-r from-gray-900/80 via-gray-800/80 to-gray-900/80 backdrop-blur-xl border border-gray-600/50 rounded-2xl p-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-green-400">99.9%</div>
                  <div className="text-gray-300 text-sm uppercase tracking-wider">Detection Rate</div>
                </div>
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-blue-400">1M+</div>
                  <div className="text-gray-300 text-sm uppercase tracking-wider">URLs Scanned</div>
                </div>
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-purple-400">24/7</div>
                  <div className="text-gray-300 text-sm uppercase tracking-wider">Protection</div>
                </div>
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-red-400">0.3s</div>
                  <div className="text-gray-300 text-sm uppercase tracking-wider">Avg Scan Time</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-xl">
          <div className="bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 border-2 border-gray-600/50 rounded-3xl max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl backdrop-blur-2xl">
            {/* Enhanced Modal Header */}
            <div className={`p-10 border-b border-gray-700 ${
              isPhishing ? 'bg-gradient-to-r from-red-900/60 to-orange-900/60' : 
              isLegitimate ? 'bg-gradient-to-r from-green-900/60 to-emerald-900/60' : 
              'bg-gradient-to-r from-gray-800/60 to-gray-700/60'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-20 h-20 bg-gradient-to-r from-red-500 to-orange-500 rounded-full flex items-center justify-center mr-8 animate-pulse shadow-2xl shadow-red-500/50">
                      <span className="text-4xl">⚠️</span>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-20 h-20 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center mr-8 animate-pulse shadow-2xl shadow-green-500/50">
                      <span className="text-4xl">✅</span>
                    </div>
                  ) : (
                    <div className="w-20 h-20 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full flex items-center justify-center mr-8 animate-pulse shadow-2xl shadow-yellow-500/50">
                      <span className="text-4xl">🔍</span>
                    </div>
                  )}
                  <div>
                    <h2 className="text-4xl font-bold text-white mb-3">QUANTUM SCAN COMPLETE</h2>
                    <p className="text-gray-300 text-xl">Advanced threat analysis results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-white transition-colors p-3 hover:bg-gray-700/50 rounded-full"
                >
                  <span className="text-3xl">✕</span>
                </button>
              </div>
            </div>

            {/* Enhanced Modal Content */}
            <div className="p-10">
              {/* Main Result */}
              <div className={`p-8 rounded-2xl mb-10 text-center border-2 ${
                isPhishing ? 'bg-gradient-to-r from-red-900/60 to-orange-900/60 border-red-500/50' : 
                isLegitimate ? 'bg-gradient-to-r from-green-900/60 to-emerald-900/60 border-green-500/50' : 
                'bg-gradient-to-r from-yellow-900/60 to-orange-900/60 border-yellow-500/50'
              }`}>
                <p className={`text-3xl font-bold mb-2 ${
                  isPhishing ? 'text-red-300' : 
                  isLegitimate ? 'text-green-300' : 
                  'text-yellow-300'
                }`}>
                  {result.result_text || 'Analysis complete'}
                </p>
              </div>

              {/* Enhanced Detailed Analysis */}
              {result.additional_info && Object.keys(result.additional_info).length > 0 && (
                <div>
                  <h3 className="text-3xl font-bold text-white mb-8 flex items-center">
                    <span className="text-cyan-400 mr-4">📊</span>
                    DETAILED QUANTUM ANALYSIS
                  </h3>
                  
                  <div className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 rounded-2xl overflow-hidden border-2 border-gray-600/50 backdrop-blur-sm">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-8">
                      {Object.entries(result.additional_info).map(([key, value], index) => (
                        <div key={index} className="bg-gradient-to-br from-gray-800/70 to-gray-900/70 rounded-xl p-6 border border-gray-700/50 backdrop-blur-sm hover:border-cyan-500/50 transition-all duration-300">
                          <div className="text-sm font-bold text-cyan-400 mb-3 uppercase tracking-widest">
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                          <div className="text-white font-mono text-xl font-semibold">
                            {value || 'N/A'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Enhanced Modal Footer */}
            <div className="p-10 border-t border-gray-700 bg-gray-800/30">
              <button
                onClick={closeModal}
                className="w-full py-6 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-500 hover:to-gray-600 text-white font-bold text-xl rounded-xl transition-all duration-300 uppercase tracking-wider shadow-xl"
              >
                🔒 CLOSE ANALYSIS REPORT
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Custom CSS for grid animation */}
      <style jsx>{`
        @keyframes moveGrid {
          0% { transform: translate(0, 0); }
          100% { transform: translate(50px, 50px); }
        }
      `}</style>
    </div>
  );
};

export default PhishingDetector;
