import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [diagnosisNote, setDiagnosisNote] = useState("");
  const [reply, setReply] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function onFileChange(event) {
    const chosen = event.target.files[0];
    setDiagnosis(null);
    setDiagnosisNote("");
    setReply("");
    setError("");
    setFile(chosen || null);
    setPreview(chosen ? URL.createObjectURL(chosen) : null);
  }

  async function fetchJson(path, formData) {
    const response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      // The caller shows the Bangla message the API returned.
      throw new Error(data.message_bn || data.message_en || data.detail || "কিছু ভুল হয়েছে।");
    }
    return data;
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("প্রথমে একটি ছবি বেছে নিন।");
      return;
    }
    setLoading(true);
    setDiagnosis(null);
    setDiagnosisNote("");
    setReply("");
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      // Run both endpoints on the same photo. Each one is independent:
      // a 503 from /diagnose (model missing) still lets the advisor reply.
      const [diagnosisResult, advisorResult] = await Promise.allSettled([
        fetchJson("/diagnose", formData),
        fetchJson("/advisor/upload", formData),
      ]);

      if (diagnosisResult.status === "fulfilled") {
        setDiagnosis(diagnosisResult.value);
      } else {
        setDiagnosisNote(diagnosisResult.reason.message);
      }

      if (advisorResult.status === "fulfilled") {
        setReply(advisorResult.value.reply_bn || "");
      } else if (diagnosisResult.status !== "fulfilled") {
        // Only show a hard error when both endpoints failed.
        setError(advisorResult.reason.message);
      }
    } catch {
      setError("সার্ভারের সঙ্গে যোগাযোগ করা যায়নি।");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Bangladesh Agriculture Intelligence</h1>
      <p>ফসলের ছবি আপলোড করুন, বাংলায় রোগ নির্ণয় ও পরামর্শ পান।</p>
      <form onSubmit={onSubmit}>
        <label>
          ফসলের ছবি
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onFileChange}
          />
        </label>
        {preview && (
          <div>
            <img src={preview} alt="Chosen crop photo" width="320" />
          </div>
        )}
        <button type="submit" disabled={loading || !file}>
          {loading ? "অপেক্ষা করুন..." : "রোগ নির্ণয় ও পরামর্শ"}
        </button>
      </form>
      {diagnosis && (
        <section>
          <h2>রোগ নির্ণয় (অফলাইন মডেল)</h2>
          <p>
            <strong>{diagnosis.disease_bn}</strong> ({diagnosis.disease_en})
          </p>
          <p>নিশ্চয়তা: {diagnosis.confidence_percent}%</p>
          <p>{diagnosis.advice_bn}</p>
        </section>
      )}
      {diagnosisNote && (
        <section>
          <h2>রোগ নির্ণয়ের বার্তা</h2>
          <p>{diagnosisNote}</p>
        </section>
      )}
      {reply && (
        <section>
          <h2>কৃষি উপদেষ্টার পরামর্শ</h2>
          <p>{reply}</p>
        </section>
      )}
      {error && (
        <section>
          <h2>বার্তা</h2>
          <p>{error}</p>
        </section>
      )}
    </main>
  );
}
