import { useEffect, useState } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

const StatisticsPage = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [recalcLoading, setRecalcLoading] = useState(false);

  const fetchStats = () => {
    setLoading(true);
    axios
      .get(`${API_URL}/statistics`)
      .then((res) => setStats(res.data))
      .catch((err) => {
        setMessage("Failed to load statistics");
        console.log(err);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleRecalculate = async () => {
    setRecalcLoading(true);
    setMessage("");
    try {
      const res = await axios.post(`${API_URL}/statistics/recalculate`);
      setStats(res.data);
    } catch (err) {
      console.log(err);
      setMessage("Tính toán lại thất bại");
    } finally {
      setRecalcLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-12 px-6 py-8 bg-white rounded-3xl shadow-lg text-center">
      <h2 className="text-3xl font-bold text-gray-800 mb-8">
        So sánh các phương pháp
      </h2>
      <button
        className="mb-6 px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
        onClick={handleRecalculate}
        disabled={recalcLoading}
      >
        {recalcLoading ? "Đang tính toán..." : "Tính toán lại"}
      </button>
      {stats && stats.last_calculated && (
        <div className="mb-4 text-gray-500 text-sm">
          Lần tính toán gần nhất:{" "}
          {new Date(stats.last_calculated).toLocaleString()}
        </div>
      )}
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
            {stats &&
              stats.results &&
              Object.entries(stats.results).map(([method, values]) => (
                <tr key={method}>
                  <td className="py-2 px-4 border font-semibold">{method}</td>
                  <td className="py-2 px-4 border">
                    {values.f1.toFixed(3)}
                  </td>
                  <td className="py-2 px-4 border">{values.map.toFixed(3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default StatisticsPage;
