import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './screens/Dashboard'
import Predict from './screens/Predict'
import Train from './screens/Train'
import Map from './screens/Map'
import Eval from './screens/Eval'
import Data from './screens/Data'
import Checkpoints from './screens/Checkpoints'
import Logs from './screens/Logs'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="predict" element={<Predict />} />
          <Route path="train" element={<Train />} />
          <Route path="map" element={<Map />} />
          <Route path="eval" element={<Eval />} />
          <Route path="data" element={<Data />} />
          <Route path="checkpoints" element={<Checkpoints />} />
          <Route path="logs" element={<Logs />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
