const axios = require('axios');

async function getPrediction(study_hours, class_attendance) {
    const response = await axios.post('http://localhost:5000/predict', {
        study_hours: study_hours,
        class_attendance: class_attendance
    });

    console.log("Probability of passing:", response.data.probability_of_passing);
}

// Example usage
getPrediction(5, 90);