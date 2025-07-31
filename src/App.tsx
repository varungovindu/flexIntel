import React, { Suspense, lazy } from 'react';
import './App.css';

// Lazy load components for better performance
const FitnessCoach = lazy(() => import('./components/FitnessCoach'));
const LoadingSpinner = lazy(() => import('./components/LoadingSpinner'));

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>flexIntel</h1>
        <p>AI-powered fitness coach using MediaPipe pose estimation</p>
      </header>
      <main>
        <Suspense fallback={<LoadingSpinner />}>
          <FitnessCoach />
        </Suspense>
      </main>
    </div>
  );
}

export default App;
