import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) {
      setMessage('Please enter a search term');
      return;
    }
  
    try {
      const response = await axios.post(`${API_URL}/get-word2vec-result`, {
        params: { query }
      });
      setResults(response.data.results);
      setMessage('');
    } catch (error) {
      setMessage(`Error searching data: ${error.message}`);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="border p-2"
        placeholder="Enter search term"
      />
      <button onClick={handleSearch} className="ml-2 btn">Search</button>
      {message && <p className="text-red-500">{message}</p>}
      <div>
        <ul>
          {results.length > 0 ? (
            results.map((result, index) => <li key={index}>{result}</li>)
          ) : (
            <li>No results found</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default SearchBar;
