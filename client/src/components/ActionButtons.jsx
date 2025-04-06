import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

const ActionButtons = () => {
  const [message, setMessage] = useState('');

  const handleCrawlData = async () => {
    try {
      const response = await axios.post(`${API_URL}/crawl-data`);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(`Error crawling data: ${error.message}`);
    }
  };

  const handleConvertData = async () => {
    try {
      const response = await axios.post(`${API_URL}/convert-to-txt`);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(`Error converting data: ${error.message}`);
    }
  };

  const handleTrainData = async () => {
    try {
      const response = await axios.post(`${API_URL}/train-model`);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(`Error training data: ${error.message}`);
    }
  };

  return (
    <div className="space-x-2">
      <button onClick={handleCrawlData} className="btn">Crawl Data</button>
      <button onClick={handleConvertData} className="btn">Convert Data</button>
      <button onClick={handleTrainData} className="btn">Train Data</button>
      <div className="mt-4">
        {message && <p>{message}</p>}
      </div>
    </div>
  );
};

export default ActionButtons;
