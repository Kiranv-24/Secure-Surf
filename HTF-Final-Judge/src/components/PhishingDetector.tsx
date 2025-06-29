
import React, { useState, useEffect, useRef } from 'react';
import { Shield, Search, AlertTriangle, X } from 'lucide-react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(new Date());
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Enhanced Binary Rain Effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const binary = "01";
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops: number[] = [];

    // Initialize drops
    for (let x = 0; x < columns; x++) {
      drops[x] = Math.random() * canvas.height;
    }

    const draw = () => {
      // Black background with slight transparency for trail effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.04)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#00ff41';
      ctx.font = fontSize + 'px monospace';

      for (let i = 0; i < drops.length; i++) {
        const text = binary[Math.floor(Math.random() * binary.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;

        // Add glow effect
        ctx.shadowColor = '#00ff41';
        ctx.shadowBlur = 10;
        ctx.fillText(text, x, y);
        ctx.shadowBlur = 0;

        // Reset drop to top randomly
        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }

        drops[i]++;
      }
    };

    const interval = setInterval(draw, 50);

    // Handle window resize
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Update current time
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
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
      setError('INVALID INPUT: URL REQUIRED');
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
      setError('SYSTEM ERROR: Failed to analyze URL. Please check your connection and try again.');
    } finally {
      setTimeout(() => {
        setLoading(false);
        setScanProgress(0);
      }, 1200);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
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

  const isPhishing = result?.result_text?.toLowerCase().includes('threat detected') || result?.result_text?.toLowerCase().includes('phishing');
  const isLegitimate = result?.result_text?.toLowerCase().includes('verified') || result?.result_text?.toLowerCase().includes('legitimate');

  return (
    <div className="min-h-screen bg-black text-green-400 relative overflow-hidden font-mono">
      {/* Enhanced Binary Rain Canvas */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-0"
        style={{ background: 'linear-gradient(45deg, #000000, #001100)' }}
      />

      {/* Cyber Grid Overlay */}
      <div className="fixed inset-0 pointer-events-none z-10 opacity-20">
        <div className="w-full h-full" style={{
          backgroundImage: `
            linear-gradient(rgba(0,255,65,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,65,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}></div>
      </div>

      {/* Main Content */}
      <div className="relative z-20 min-h-screen flex flex-col">
        {/* Top Status Bar */}
        <div className="w-full bg-black/90 border-b border-green-400/30 p-4 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto flex justify-between items-center text-sm">
            <div className="flex items-center space-x-6">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2"></div>
                <span>NEURAL_NET_ONLINE</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-2"></div>
                <span>THREAT_DB_UPDATED</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse mr-2"></div>
                <span>REAL_TIME_PROTECTION</span>
              </div>
            </div>
            <div className="text-green-400/70">
              {currentTime.toLocaleTimeString()} | SYSTEM_OPERATIONAL
            </div>
          </div>
        </div>

        {/* Main Interface */}
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="w-full max-w-4xl">
            
            {/* Cyber Header with Enhanced Design */}
            <div className="text-center mb-8">
              <div className="relative inline-block mb-6">
                <div className="absolute inset-0 bg-green-400/20 blur-2xl rounded-full animate-pulse"></div>
                <div className="relative bg-black/80 border-2 border-green-400 rounded-full p-6 backdrop-blur-sm">
                  <Shield className="w-16 h-16 text-green-400 animate-pulse" />
                </div>
              </div>
              
              <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-blue-400 to-red-400 bg-clip-text text-transparent animate-pulse">
                CYBER GUARDIAN
              </h1>
              
              <div className="text-xl text-green-400/80 mb-6 tracking-wider">
                ADVANCED PHISHING DETECTION SYSTEM
              </div>

              <div className="flex justify-center space-x-8 text-xs">
                {/* <div className="bg-black/60 border border-green-400/30 rounded px-4 py-2">
                  <div className="text-green-400 font-bold">AI ENGINE</div>
                  <div className="text-green-400/70">v2.4.7</div>
                </div>
                <div className="bg-black/60 border border-blue-400/30 rounded px-4 py-2">
                  <div className="text-blue-400 font-bold">THREAT DB</div>
                  <div className="text-blue-400/70">15.2M ENTRIES</div>
                </div>
                <div className="bg-black/60 border border-red-400/30 rounded px-4 py-2">
                  <div className="text-red-400 font-bold">SCAN RATE</div>
                  <div className="text-red-400/70">99.97%</div>
                </div> */}
              </div>
            </div>

            {/* Enhanced Scanner Interface */}
            <div className="bg-black/80 border-2 border-green-400/30 rounded-lg backdrop-blur-sm shadow-2xl shadow-green-400/20 overflow-hidden">
              
              {/* Terminal Header */}
              <div className="bg-gradient-to-r from-green-900/20 to-blue-900/20 p-4 border-b border-green-400/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                      <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-green-400 font-bold">root@cyberguardian:~$ threat_scanner.exe</span>
                  </div>
                  <div className="text-green-400/70 text-sm">
                    SESSION: {Date.now().toString().slice(-6)}
                  </div>
                </div>
              </div>

              <div className="p-6">
                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* URL Input Section */}
                  <div>
                    <div className="flex items-center mb-3">
                      <span className="text-green-400 font-bold mr-3">TARGET_URL:</span>
                      <div className="w-2 h-4 bg-green-400 animate-pulse"></div>
                    </div>
                    <div className="relative">
                      <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://suspicious-domain.com"
                        className="w-full px-6 py-4 bg-black/70 border-2 border-green-400/50 hover:border-green-400 focus:border-green-400 rounded-lg text-green-400 placeholder-green-400/50 focus:outline-none font-mono backdrop-blur-sm transition-all duration-300 text-lg shadow-inner"
                        disabled={loading}
                      />
                      {url && (
                       <button
                          type="button"
                          onClick={clearInput}
                          aria-label="Clear input"
                          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-green-400/70 hover:text-red-400 transition-colors p-1"
                        >
                          <X className="w-5 h-5" />
                        </button>

                      )}
                    </div>
                  </div>

                  {/* Enhanced Scan Button */}
                  <button
                    type="submit"
                    disabled={loading || !url.trim()}
                    className="w-full py-4 bg-gradient-to-r from-green-600 via-blue-600 to-red-600 hover:from-green-500 hover:via-blue-500 hover:to-red-500 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed text-black font-bold text-lg rounded-lg transition-all duration-300 border-2 border-green-400/50 hover:border-green-400 shadow-lg hover:shadow-green-400/30 relative overflow-hidden group"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-red-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    <div className="relative flex items-center justify-center space-x-3">
                      {loading ? (
                        <>
                          <div className="animate-spin rounded-full h-6 w-6 border-3 border-black border-t-transparent"></div>
                          <span>DEEP_SCAN_IN_PROGRESS</span>
                        </>
                      ) : (
                        <>
                          <Search className="w-6 h-6" />
                          <span>INITIATE_THREAT_SCAN</span>
                        </>
                      )}
                    </div>
                  </button>

                  {/* Enhanced Progress Section */}
                  {loading && (
                    <div className="space-y-4 bg-black/50 rounded-lg p-4 border border-green-400/30">
                      <div className="flex justify-between items-center">
                        <span className="text-green-400 font-bold">NEURAL_ANALYSIS_PROGRESS</span>
                        <span className="text-green-400 font-mono">{Math.round(scanProgress)}%</span>
                      </div>
                      
                      <div className="relative h-4 bg-gray-800 rounded-lg overflow-hidden border border-green-400/30">
                        <div 
                          className="h-full bg-gradient-to-r from-green-400 via-blue-400 to-red-400 transition-all duration-300 relative"
                          style={{ width: `${scanProgress}%` }}
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span>URL_PARSING:</span>
                            <span className="text-green-400">COMPLETE</span>
                          </div>
                          <div className="flex justify-between">
                            <span>DNS_LOOKUP:</span>
                            <span className="text-green-400">COMPLETE</span>
                          </div>
                          <div className="flex justify-between">
                            <span>REPUTATION_CHECK:</span>
                            <span className="text-yellow-400">PROCESSING</span>
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span>CONTENT_ANALYSIS:</span>
                            <span className="text-yellow-400">PROCESSING</span>
                          </div>
                          <div className="flex justify-between">
                            <span>ML_CLASSIFICATION:</span>
                            <span className="text-gray-400">PENDING</span>
                          </div>
                          <div className="flex justify-between">
                            <span>FINAL_VERDICT:</span>
                            <span className="text-gray-400">PENDING</span>
                          </div>
                        </div>
                      </div>

                      <div className="text-center">
                        <div className="text-green-400/70 animate-pulse">
                          🔍 Analyzing threat patterns and behavioral signatures...
                        </div>
                      </div>
                    </div>
                  )}
                </form>

                {/* Error Display */}
                {error && (
                  <div className="mt-4 p-4 bg-red-900/30 border-2 border-red-500/50 rounded-lg flex items-center backdrop-blur-sm">
                    <AlertTriangle className="text-red-400 mr-3 w-6 h-6" />
                    <span className="text-red-300 font-bold">{error}</span>
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
          <div className="bg-black/90 border-2 border-green-400/50 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl shadow-green-400/30">
            
            {/* Modal Header */}
            <div className={`p-6 border-b-2 border-green-400/30 ${
              isPhishing ? 'bg-gradient-to-r from-red-900/30 to-red-800/20' : 
              isLegitimate ? 'bg-gradient-to-r from-green-900/30 to-green-800/20' : 
              'bg-gradient-to-r from-gray-800/30 to-gray-700/20'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`w-16 h-16 rounded-lg flex items-center justify-center mr-4 border-2 ${
                    isPhishing ? 'bg-red-500/20 border-red-500' : 
                    isLegitimate ? 'bg-green-500/20 border-green-500' : 
                    'bg-yellow-500/20 border-yellow-500'
                  }`}>
                    {isPhishing ? (
                      <AlertTriangle className="w-8 h-8 text-red-400" />
                    ) : isLegitimate ? (
                      <Shield className="w-8 h-8 text-green-400" />
                    ) : (
                      <Search className="w-8 h-8 text-yellow-400" />
                    )}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-green-400">SCAN_COMPLETE</h2>
                    <p className="text-green-400/70">Threat analysis finished</p>
                  </div>
                </div>
                <button
                onClick={closeModal}
                type="button"
                aria-label="Close modal"
                className="text-green-400/70 hover:text-red-400 transition-colors p-2"
              >
                <X className="w-8 h-8" />
              </button>

              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6 max-h-[60vh] overflow-y-auto">
              {/* Main Result */}
              <div className={`p-6 rounded-lg border-2 mb-6 ${
                isPhishing ? 'bg-red-900/20 border-red-500/50' : 
                isLegitimate ? 'bg-green-900/20 border-green-500/50' : 
                'bg-yellow-900/20 border-yellow-500/50'
              }`}>
                <div className="text-center">
                  <div className="text-sm text-green-400/70 mb-2">FINAL_VERDICT:</div>
                  <div className={`text-2xl font-bold mb-4 ${
                    isPhishing ? 'text-red-400' : 
                    isLegitimate ? 'text-green-400' : 
                    'text-yellow-400'
                  }`}>
                    {result.result_text || 'ANALYSIS_COMPLETE'}
                  </div>
                  
                  {result.additional_info?.neural_confidence && (
                    <div className="text-green-400/80">
                      Neural Network Confidence: {result.additional_info.neural_confidence}
                    </div>
                  )}
                </div>
              </div>

              {/* Detailed Analysis */}
              {result.additional_info && Object.keys(result.additional_info).length > 0 && (
                <div>
                  <h3 className="text-green-400 font-bold text-xl mb-4 flex items-center">
                    <span className="mr-3">📊</span>
                    DETAILED_ANALYSIS_REPORT
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(result.additional_info).map(([key, value], index) => (
                      <div key={index} className="bg-black/60 border border-green-400/30 rounded-lg p-4">
                        <div className="text-green-400 text-sm font-bold uppercase mb-2">
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className={`text-lg font-mono ${
                          key === 'threat_level' && value === 'HIGH' ? 'text-red-400' :
                          key === 'threat_level' && value === 'LOW' ? 'text-green-400' :
                          key === 'ssl_certificate' && value === 'VALID' ? 'text-green-400' :
                          key === 'ssl_certificate' && value === 'INVALID' ? 'text-red-400' :
                          'text-green-400/80'
                        }`}>
                          {String(value) || 'NULL'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t-2 border-green-400/30 bg-black/50">
              <div className="flex space-x-4">
                <button
                  onClick={closeModal}
                  className="flex-1 py-3 bg-green-600 hover:bg-green-500 text-black font-bold rounded-lg transition-all duration-300 border-2 border-green-400/50"
                >
                  CLOSE_ANALYSIS
                </button>
                <button
                  onClick={() => {
                    setShowModal(false);
                    setUrl('');
                    setResult(null);
                  }}
                  className="flex-1 py-3 bg-blue-600 hover:bg-blue-500 text-black font-bold rounded-lg transition-all duration-300 border-2 border-blue-400/50"
                >
                  NEW_SCAN
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhishingDetector;
