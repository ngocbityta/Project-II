import { useState } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

const ActionButtons = () => {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState("");

  const handleAction = async (actionName, endpoint) => {
    setLoading(actionName);
    setMessage("");

    try {
      const response = await axios.post(`${API_URL}/${endpoint}`);
      const data = response.data;

      if (data.output?.status === "fail") {
        setMessage(
          `${actionName} failed: ${data.output.error || "Unknown error"}`
        );
      } else {
        setMessage(`✅ ${data.message}`);
      }
    } catch (error) {
      setMessage(`Error during ${actionName.toLowerCase()}: ${error.message}`);
    } finally {
      setLoading("");
    }
  };

  const actions = [
    {
      label: "Crawl Data",
      action: "Crawling",
      endpoint: "crawl-data",
      bg: "bg-green-600",
      hover: "hover:bg-green-700",
    },
    {
      label: "Convert Data",
      action: "Converting",
      endpoint: "convert-to-txt",
      bg: "bg-yellow-500",
      hover: "hover:bg-yellow-600",
    },
    {
      label: "Train Word2Vec",
      action: "Training Word2Vec Model",
      endpoint: "train-word2vec-model",
      bg: "bg-blue-600",
      hover: "hover:bg-blue-700",
    },
    {
      label: "Train TF-IDF",
      action: "Training TF-IDF Model",
      endpoint: "train-tfidf-model",
      bg: "bg-orange-600",
      hover: "hover:bg-orange-700",
    },
    {
      label: "Train Doc2Vec (DBOW)",
      action: "Training Doc2Vec DBOW",
      endpoint: "train-doc2vec-dbow",
      bg: "bg-purple-600",
      hover: "hover:bg-purple-700",
    },
  ];

  return (
    <div className="max-w-3xl mx-auto mt-12 px-6 py-8 bg-white rounded-3xl shadow-lg text-center">
      <h2 className="text-3xl font-bold text-gray-800 mb-8">Data Processing Actions</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 gap-5">
        {actions.map(({ label, action, endpoint, bg, hover }) => (
          <button
            key={action}
            onClick={() => handleAction(action, endpoint)}
            className={`text-white px-6 py-3 rounded-xl transition duration-200 flex items-center justify-center disabled:opacity-50 ${bg} ${hover}`}
            disabled={loading === action}
          >
            {loading === action ? <Spinner /> : label}
          </button>
        ))}
      </div>

      {message && (
        <div className="mt-6 px-4 py-3 bg-gray-100 rounded-xl text-gray-700 text-sm font-medium border border-gray-200">
          {message}
        </div>
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
