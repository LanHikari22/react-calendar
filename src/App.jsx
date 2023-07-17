import React, { useEffect } from 'react';
import './App.css';
import Calendar from './components/Calendar';

const App = () => {
  useEffect(() => {
    const interval = setInterval(() => {
      fetch('/markers.json') // assuming the markers.json file is in the public folder
        .then(response => response.json())
        .then(newMarkers => {
          const localStorageMarkers = localStorage.getItem('markers');
          if (localStorageMarkers !== JSON.stringify(newMarkers['markers'])) {
            localStorage.setItem('markers', JSON.stringify(newMarkers['markers']));
            console.log('updating markers...', localStorage.getItem('markers').length, JSON.stringify(newMarkers['markers']).length);
            // Perform any additional actions here, such as updating state or triggering component re-renders
            this.setState({});
          }
        })
        .catch(error => {
          // Handle error if the file cannot be fetched
          console.error('Error fetching markers.json:', error);
        });
    }, 5000); // Interval duration in milliseconds (e.g., 5000ms = 5 seconds)

    return () => {
      clearInterval(interval); // Clean up the interval when the component unmounts
    };
  }, []);

  return (
    <div className="App">
      <Calendar />
    </div>
  );
};

export default App;

//In this modified version, the useEffect hook is added to the App component. Inside the useEffect hook, the periodic event is set up to fetch and compare the contents of the markers.json file with the value stored in the markers key of the localStorage. If they differ, it updates the localStorage with the new value.

//Please note that you need to make sure the markers.json file is located in the public folder of your React app, and adjust the fetch URL if necessary.

