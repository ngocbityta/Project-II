import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [word2vecResults, setWord2vecResults] = useState([]);
  const [doc2vecResults, setDoc2vecResults] = useState([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) {
      setMessage('Please enter a search term');
      return;
    }
  
    setLoading(true);
    setMessage('');
    setWord2vecResults([]);
    setDoc2vecResults([]);
  
    try {
      const [word2vecRes, doc2vecRes] = await Promise.all([
        axios.post(`${API_URL}/get-word2vec-result`, { sentence: query }),
        axios.post(`${API_URL}/get-doc2vec-result`, { sentence: query }),
      ]);
  
      if (word2vecRes.data.output?.similarities) {
        setWord2vecResults(word2vecRes.data.output.similarities);
      }

  
      if (doc2vecRes.data.output?.similarities) {
        setDoc2vecResults(doc2vecRes.data.output.similarities);
      }
  
      if (
        !word2vecRes.data.output?.similarities?.length &&
        !doc2vecRes.data.output?.similarities?.length
      ) {
        setMessage('No results found');
      }
    } catch (error) {
      setMessage(`Error searching data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  

  const renderResults = (results, title) => (
    <div className="w-full md:w-1/2">
      <h3 className="text-lg font-semibold mb-2 text-center text-gray-700">{title}</h3>
      {results.length > 0 ? (
        <ul className="space-y-2">
          {results.map((result, index) => (
            <li
              key={index}
              className="bg-gray-100 px-4 py-2 rounded-lg shadow-sm hover:bg-gray-200 transition"
            >
              <p className="font-medium text-gray-700">{result.sentence}</p>
              <p className="text-sm text-gray-500">
                Cosine similarity: {result.cosine_similarity.toFixed(3)}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        !loading && <p className="text-gray-500 text-center">No results</p>
      )}
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto p-4 bg-white rounded-2xl shadow-md mt-10">
      <h2 className="text-2xl font-semibold mb-4 text-center text-gray-800">Search Word2Vec & TF-IDF</h2>
      <div className="flex flex-col sm:flex-row gap-2 items-center">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
          placeholder="Enter search term"
        />
        <button
          onClick={handleSearch}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition duration-200 flex items-center justify-center min-w-[100px]"
          disabled={loading}
        >
          {loading ? (
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
          ) : (
            'Search'
          )}
        </button>
      </div>

      {message && (
        <p className="text-red-500 mt-3 text-center font-medium">{message}</p>
      )}

      <div className="mt-6 flex flex-col md:flex-row gap-6">
        {renderResults(word2vecResults, 'Word2Vec Results')}
        {renderResults(doc2vecResults, 'Doc2Vec Results')}
      </div>
    </div>
  );
};

export default SearchBar;
