import Dashboard from './pages/Dashboard'
import AIModels from './pages/AIModels'
import { RegionProvider } from './context/RegionContext'
import { ShelterProvider } from './context/ShelterContext'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

function App() {
  return (
    <BrowserRouter>
      <RegionProvider>
        <ShelterProvider>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<AIModels />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </ShelterProvider>
      </RegionProvider>
    </BrowserRouter>
  )
}

export default App
