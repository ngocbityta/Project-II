import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DataActionsPage from './pages/DataActionsPage';
import SearchPage from './pages/SearchPage';

const AppRoutes = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<DataActionsPage />} />
      <Route path="/search" element={<SearchPage />} />
    </Routes>
  </BrowserRouter>
);

export default AppRoutes;
