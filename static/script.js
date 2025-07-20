document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();

    if (data.error) {
        document.getElementById('result').innerText = `Error: ${data.error}`;
    } else {
        document.getElementById('result').innerText = 
            `Predicted Estimated Salary: ₹ ${data.predicted_salary.toFixed(2)}`;
    }
});
