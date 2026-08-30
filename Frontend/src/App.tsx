import { useState } from 'react'
import './App.css'

// Define the expected prediction structure based on the API contract[cite: 1]
interface PredictionResult {
  success: boolean;
  prediction: {
    class_english: string;
    class_italian: string;
    confidence: number;
    message: string;
  };
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  // Handle Image Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null) // Reset previous prediction
    }
  }

  // Handle API Request
  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    
    // Creating the strict multipart/form-data structure required by Backend[cite: 1]
    const formData = new FormData();
    formData.append("file", file); 

    try {
      // Hitting the local Mock Backend[cite: 1]
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error connecting to Backend:", error);
      alert("Backend se connect nahi hua. Kya server on hai?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Animal Classifier 🐾</h1>
      
      <div className="card">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        
        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Upload Preview" />
          </div>
        )}

        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? "Analyzing..." : "Detect Animal"}
        </button>
      </div>

      {/* Rendering the Locked JSON Structure from Backend[cite: 1] */}
      {result && result.success && (
        <div className="result-card">
          <h2>Result: {result.prediction.class_english.toUpperCase()}</h2>
          <p><strong>Italian Label:</strong> {result.prediction.class_italian}</p>
          <p><strong>Confidence:</strong> {(result.prediction.confidence * 100).toFixed(1)}%</p>
          <p className="success-msg">{result.prediction.message}</p>
        </div>
      )}
    </div>
  )
}

export default App