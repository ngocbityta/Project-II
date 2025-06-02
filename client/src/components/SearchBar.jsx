import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [word2vecResults, setWord2vecResults] = useState([]);
  const [doc2vecResults, setDoc2vecResults] = useState([]);
  const [word2vecAccuracy, setWord2vecAccuracy] = useState(null);
  const [doc2vecAccuracy, setDoc2vecAccuracy] = useState(null);
  const [tfidfResults, setTfidfResults] = useState([]);
  const [tfidfAccuracy, setTfidfAccuracy] = useState(null);
  const [bm25Results, setBm25Results] = useState([]);
  const [bm25Accuracy, setBm25Accuracy] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [bertResults, setBertResults] = useState([]);
  const [bertAccuracy, serBertAccuraty] = useState(null);
  const [tfidfBertResults, setTfidfBertResults] = useState([]);
  const [tfidfBertAccuracy, setTfidfBertAccuracy] = useState(null); // Thêm state cho F1 Score

  const handleSearch = async () => {
    if (!query.trim()) {
      setMessage('Please enter a search term');
      return;
    }

    setLoading(true);
    setMessage('');
    setWord2vecResults([]);
    setDoc2vecResults([]);
    setWord2vecAccuracy(null);
    setDoc2vecAccuracy(null);
    setTfidfResults([]);
    setTfidfAccuracy(null);
    setBm25Results([]);
    setBm25Accuracy(null);
    setBertResults([]);
    serBertAccuraty(null);
    setTfidfBertResults([]);
    setTfidfBertAccuracy(null); // Reset F1 Score

    try {
      const [word2vecRes, doc2vecRes, tfidfRes, bm25Res, bertRes] = await Promise.all([
        axios.post(`${API_URL}/get-word2vec-result`, { sentence: query }),
        axios.post(`${API_URL}/get-doc2vec-result`, { sentence: query }),
        axios.post(`${API_URL}/get-tfidf-result`, { sentence: query }),
        axios.post(`${API_URL}/get-bm25-result`, { sentence: query }),
        axios.post(`${API_URL}/get-bert-result`, { sentence: query }),
      ]);

      if (word2vecRes.data.output?.similarities) {
        setWord2vecResults(word2vecRes.data.output.similarities);
        setWord2vecAccuracy(word2vecRes.data.output.accuracy ?? null);
      }

      if (doc2vecRes.data.output?.similarities) {
        setDoc2vecResults(doc2vecRes.data.output.similarities);
        setDoc2vecAccuracy(doc2vecRes.data.output.accuracy ?? null);
      } 
      if (tfidfRes.data.output?.similarities) {
        setTfidfResults(tfidfRes.data.output.similarities);
        setTfidfAccuracy(tfidfRes.data.output.accuracy ?? null);
      }

      if (bm25Res.data.output?.similarities) {
        setBm25Results(bm25Res.data.output.similarities);
        setBm25Accuracy(bm25Res.data.output.accuracy ?? null);
      }

      if (bertRes.data.output?.similarities) {
        setBertResults(bertRes.data.output.similarities);
        serBertAccuraty(bertRes.data.output.accuracy ?? null);
      }

      // TF-IDF+BERT
      if (tfidfRes.data.output?.similarities && bertRes.data.output?.similarities) {
        const alpha = 0.7;
        const tfidfDict = {};
        tfidfRes.data.output.similarities.forEach(item => {
          const sent = (item.sentence ?? '').trim();
          tfidfDict[sent] = typeof item.score === 'number' ? item.score : (item.cosine_similarity ?? 0);
        });
        const bertDict = {};
        bertRes.data.output.similarities.forEach(item => {
          const sent = (item.sentence ?? '').trim();
          bertDict[sent] = typeof item.score === 'number' ? item.score : (item.cosine_similarity ?? 0);
        });
        const allSentences = Array.from(new Set([...Object.keys(tfidfDict), ...Object.keys(bertDict)]));
        const combined = allSentences.map(sent => {
          const tfidf_score = tfidfDict[sent] ?? 0;
          const bert_score = bertDict[sent] ?? 0;
          const final_score = alpha * tfidf_score + (1 - alpha) * bert_score;
          return {
            sentence: sent,
            final_score,
          };
        });
        combined.sort((a, b) => b.final_score - a.final_score);
        // Lấy top 20 câu (không hiện điểm)
        const topCombined = combined.slice(0, 20);
        setTfidfBertResults(topCombined);

        // Tính F1 Score cho TF-IDF+BERT
        // Gọi lại API get-tfidf-bert-result để lấy accuracy (F1)
        try {
          const tfidfBertRes = await axios.post(`${API_URL}/get-tfidf-bert-result`, { sentence: query });
          setTfidfBertAccuracy(tfidfBertRes.data.output?.accuracy ?? null);
        } catch {
          setTfidfBertAccuracy(null);
        }
      }

      if (
        !word2vecRes.data.output?.similarities?.length &&
        !doc2vecRes.data.output?.similarities?.length &&
        !tfidfRes.data.output?.similarities?.length &&
        !bm25Res.data.output?.similarities?.length &&
        !bertRes.data.output?.similarities?.length
      ) {
        setMessage('No results found');
      }
    } catch (error) {
      setMessage(`Error searching data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderResults = (results, title, accuracy) => (
    <div className="w-full md:w-1/2">
      <h3 className="text-lg font-semibold mb-2 text-center text-gray-700">{title}</h3>
      {accuracy !== null && accuracy !== undefined && (
        <p className="text-center text-sm text-green-600 font-medium mb-2">
          F1 Score: {(accuracy * 100).toFixed(2)}
        </p>
      )}
      {results.length > 0 ? (
        <ul className="space-y-2">
          {results.map((result, index) => (
            <li
              key={index}
              className="bg-gray-100 px-4 py-2 rounded-lg shadow-sm hover:bg-gray-200 transition"
            >
              <p className="font-medium text-gray-700">{result.sentence}</p>
            </li>
          ))}
        </ul>
      ) : (
        !loading && <p className="text-gray-500 text-center">No results</p>
      )}
    </div>
  );

  // Cột TF-IDF+BERT: chỉ hiển thị danh sách câu, có F1 Score
  const renderTfidfBertResults = (results, accuracy) => (
    <div className="w-full md:w-1/2">
      <h3 className="text-lg font-semibold mb-2 text-center text-gray-700">TF-IDF + BERT</h3>
      {accuracy !== null && accuracy !== undefined && (
        <p className="text-center text-sm text-green-600 font-medium mb-2">
          F1 Score: {(accuracy * 100).toFixed(2)}
        </p>
      )}
      {results.length > 0 ? (
        <ul className="space-y-2">
          {results.map((item, idx) => (
            <li
              key={idx}
              className="bg-gray-100 px-4 py-2 rounded-lg shadow-sm hover:bg-gray-200 transition"
            >
              <p className="font-medium text-gray-700">{item.sentence}</p>
            </li>
          ))}
        </ul>
      ) : (
        !loading && <p className="text-gray-500 text-center">No results</p>
      )}
    </div>
  );

  return (
    <div className="w-full mx-auto p-4 bg-white rounded-2xl shadow-md mt-10">
      <h2 className="text-2xl font-semibold mb-4 text-center text-gray-800">
        Search Bar
      </h2>
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
        {renderResults(word2vecResults, 'Word2Vec Results', word2vecAccuracy)}
        {renderResults(doc2vecResults, 'Doc2Vec Results', doc2vecAccuracy)}
        {renderResults(tfidfResults, 'TF-IDF Results', tfidfAccuracy)}
        {renderResults(bm25Results, 'BM25 Results', bm25Accuracy)}
        {renderResults(bertResults, 'Bert Results', bertAccuracy)}
        {renderTfidfBertResults(tfidfBertResults, tfidfBertAccuracy)}
      </div>
    </div>
  );
};

export default SearchBar;
