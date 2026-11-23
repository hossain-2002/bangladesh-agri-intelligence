import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [reply, setReply] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function onFileChange(event) {
    const chosen = event.target.files[0];
    setReply("");
    setError("");
    setFile(chosen || null);
    setPreview(chosen ? URL.createObjectURL(chosen) : null);
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("প্রথমে একটি ছবি বেছে নিন।");
      return;
    }
    setLoading(true);
    setReply("");
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/advisor/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.message_bn || data.message_en || "কিছু ভুল হয়েছে।");
        return;
      }
      setReply(data.reply_bn || "");
    } catch {
      setError("সার্ভারের সঙ্গে যোগাযোগ করা যায়নি।");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Bangladesh Agriculture Intelligence</h1>
      <p>ফসলের ছবি আপলোড করুন, বাংলায় পরামর্শ পান।</p>
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
          {loading ? "অপেক্ষা করুন..." : "পরামর্শ নিন"}
        </button>
      </form>
      {reply && (
        <section>
          <h2>পরামর্শ</h2>
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
