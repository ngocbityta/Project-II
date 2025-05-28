import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DataActionPage from './pages/DataActionPage';
import SearchPage from './pages/SearchPage';
import StatisticsPage from './pages/StatisticsPage';

const AppRoutes = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<DataActionPage />} />
      <Route path="/search" element={<SearchPage />} />
      <Route path="/statistics" element={<StatisticsPage />} />
    </Routes>
  </BrowserRouter>
);

export default AppRoutes;
