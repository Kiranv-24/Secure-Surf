
import React, { useState } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);

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
      
      setResult(response.data);
      setShowModal(true);
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
    setShowModal(false);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const isPhishing = result?.result_text?.toLowerCase().includes('phishing');
  const isLegitimate = result?.result_text?.toLowerCase().includes('legitimate');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 text-white relative overflow-hidden">
      {/* Cyber Grid Background */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}></div>
      </div>

      {/* Header */}
      <div className="relative z-10 text-center pt-16 pb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full mb-6 shadow-lg shadow-cyan-500/25">
          <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
          </svg>
        </div>
        <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent mb-4">
          SECURE-SURF
        </h1>
        <p className="text-xl text-gray-300 mb-2">Advanced Phishing Detection System</p>
        <div className="flex items-center justify-center space-x-4 text-sm text-gray-400">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
            AI-Powered Analysis
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
            Real-time Protection
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-4xl mx-auto px-6">
        {/* Search Section */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-8 mb-8 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative">
              <label htmlFor="url-input" className="block text-sm font-medium text-gray-300 mb-2">
                Enter URL to Analyze
              </label>
              <div className="relative">
                <input
                  id="url-input"
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="w-full px-4 py-4 bg-gray-900/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-300"
                  disabled={loading}
                />
                {url && (
                  <button
                    type="button"
                    onClick={clearInput}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-all duration-300 transform hover:scale-[1.02] disabled:scale-100 shadow-lg"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-2"></div>
                  Analyzing URL...
                </div>
              ) : (
                'Scan for Threats'
              )}
            </button>
          </form>

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-red-900/50 border border-red-600 rounded-xl flex items-center">
              <svg className="w-5 h-5 text-red-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="text-red-300">{error}</span>
            </div>
          )}
        </div>

        {/* Recent Threats Table */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <svg className="w-6 h-6 text-red-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Recent Malicious Websites
          </h3>
          <div className="overflow-hidden rounded-xl border border-gray-600">
            <table className="w-full">
              <thead className="bg-red-900/30">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-red-300">Threat URL</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-red-300">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {[
                  'http://payment.dev.cruzotec.com/',
                  'http://gb-apple.web.app/',
                  'https://garbare.ru/',
                  'http://swisscome.blogspot.sn/',
                  'http://ekoakyanimda.vercel.app/',
                  'https://joethenuke.github.io/account'
                ].map((threat, index) => (
                  <tr key={index} className="hover:bg-gray-700/30 transition-colors">
                    <td className="px-6 py-4 text-sm text-gray-300 font-mono">{threat}</td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-900/50 text-red-300 border border-red-600">
                        <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                        Malicious
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      {showModal && result && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            {/* Modal Header */}
            <div className={`p-6 border-b border-gray-700 ${isPhishing ? 'bg-red-900/30' : isLegitimate ? 'bg-green-900/30' : 'bg-gray-800/50'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {isPhishing ? (
                    <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center mr-4">
                      <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                  ) : isLegitimate ? (
                    <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mr-4">
                      <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                  ) : (
                    <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center mr-4">
                      <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                    </div>
                  )}
                  <div>
                    <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
                    <p className="text-gray-300">Security assessment results</p>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {/* Main Result */}
              <div className={`p-6 rounded-xl mb-6 ${
                isPhishing ? 'bg-red-900/50 border border-red-600' : 
                isLegitimate ? 'bg-green-900/50 border border-green-600' : 
                'bg-yellow-900/50 border border-yellow-600'
              }`}>
                <p className={`text-lg font-semibold ${
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
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                    <svg className="w-5 h-5 text-cyan-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" clipRule="evenodd" />
                    </svg>
                    Detailed Analysis
                  </h3>
                  
                  <div className="bg-gray-800/50 rounded-xl overflow-hidden border border-gray-700">
                    <table className="w-full">
                      <thead className="bg-cyan-900/30">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-cyan-300">Property</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-cyan-300">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-700">
                        {Object.entries(result.additional_info).map(([key, value], index) => (
                          <tr key={index} className="hover:bg-gray-700/30 transition-colors">
                            <td className="px-4 py-3 text-sm font-medium text-gray-300">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-400 font-mono">
                              {value || 'N/A'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-gray-700 bg-gray-800/30">
              <button
                onClick={closeModal}
                className="w-full py-3 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white font-semibold rounded-xl transition-all duration-300"
              >
                Close Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhishingDetector;
