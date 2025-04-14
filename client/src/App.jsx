import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DataActionPage from './pages/DataActionPage';
import SearchPage from './pages/SearchPage';

const AppRoutes = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<DataActionPage />} />
      <Route path="/search" element={<SearchPage />} />
    </Routes>
  </BrowserRouter>
);

export default AppRoutes;
