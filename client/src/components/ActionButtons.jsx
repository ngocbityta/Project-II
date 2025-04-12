import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const ActionButtons = () => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState('');

  const handleAction = async (actionName, endpoint) => {
    setLoading(actionName);
    setMessage('');

    try {
      const response = await axios.post(`${API_URL}/${endpoint}`);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(`Error during ${actionName.toLowerCase()}: ${error.message}`);
    } finally {
      setLoading('');
    }
  };

  return (
    <div className="max-w-xl mx-auto p-4 bg-white rounded-2xl shadow-md mt-10 text-center">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Data Actions</h2>
      <div className="flex flex-col sm:flex-row justify-center items-center gap-3">
        <button
          onClick={() => handleAction('Crawling', 'crawl-data')}
          className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition duration-200 w-40 flex items-center justify-center"
          disabled={loading === 'Crawling'}
        >
          {loading === 'Crawling' ? (
            <Spinner />
          ) : (
            'Crawl Data'
          )}
        </button>

        <button
          onClick={() => handleAction('Converting', 'convert-to-txt')}
          className="bg-yellow-500 text-white px-4 py-2 rounded-lg hover:bg-yellow-600 transition duration-200 w-40 flex items-center justify-center"
          disabled={loading === 'Converting'}
        >
          {loading === 'Converting' ? (
            <Spinner />
          ) : (
            'Convert Data'
          )}
        </button>

        <button
          onClick={() => handleAction('Training', 'train-model')}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition duration-200 w-40 flex items-center justify-center"
          disabled={loading === 'Training'}
        >
          {loading === 'Training' ? (
            <Spinner />
          ) : (
            'Train Data'
          )}
        </button>
      </div>

      {message && (
        <p className="mt-4 text-sm text-gray-700 font-medium bg-gray-100 p-2 rounded-lg">
          {message}
        </p>
      )}
    </div>
  );
};

const Spinner = () => (
  <svg
    className="animate-spin h-5 w-5 text-white"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z"
    />
  </svg>
);

export default ActionButtons;
