import { useEffect, useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const StatisticsPage = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get(`${API_URL}/statistics`)
      .then(res => setStats(res.data))
      .catch(err => setMessage('Failed to load statistics'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-3xl mx-auto mt-12 px-6 py-8 bg-white rounded-3xl shadow-lg text-center">
      <h2 className="text-3xl font-bold text-gray-800 mb-8">So sánh các phương pháp</h2>
      {loading ? (
        <p>Loading...</p>
      ) : message ? (
        <p className="text-red-500">{message}</p>
      ) : (
        <table className="min-w-full border border-gray-300 rounded-xl">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border">Phương pháp</th>
              <th className="py-2 px-4 border">F1 trung bình</th>
              <th className="py-2 px-4 border">MAP</th>
            </tr>
          </thead>
          <tbody>
            {stats && Object.entries(stats).map(([method, values]) => (
              <tr key={method}>
                <td className="py-2 px-4 border font-semibold">{method}</td>
                <td className="py-2 px-4 border">{(values.f1 * 100).toFixed(2)}</td>
                <td className="py-2 px-4 border">{(values.map * 100).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default StatisticsPage;
