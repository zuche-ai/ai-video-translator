import React, { useState, useEffect } from 'react';
import './App.css';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'zh', label: 'Chinese' },
];

const AUDIO_MODES = [
  { value: 'subtitles-only', label: 'Subtitles Only' },
  { value: 'replace', label: 'Replace Audio (Voice Cloning)' },
  { value: 'overlay', label: 'Overlay (Voice + Original)' },
];

function App() {
  // API base URL - use nginx proxy for all environments
  const API_BASE = '/api';
    
  const [files, setFiles] = useState([]);
  const [srcLang, setSrcLang] = useState('en');
  const [tgtLang, setTgtLang] = useState('es');
  const [audioMode, setAudioMode] = useState('subtitles-only');
  const [voiceClone, setVoiceClone] = useState(false);
  const [addCaptions, setAddCaptions] = useState(false);
  const [captionFontSize, setCaptionFontSize] = useState(24);
  const [originalVolume, setOriginalVolume] = useState(0.3);
  const [status, setStatus] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Reset isProcessing when all jobs are done or error
  useEffect(() => {
    if (status.length > 0 && status.every(s => s.state === 'Done' || s.state === 'Error')) {
      setIsProcessing(false);
    }
  }, [status]);

  // Reset isProcessing when there are no files (e.g., on page load or after removing all files)
  useEffect(() => {
    if (files.length === 0) {
      setIsProcessing(false);
    }
  }, [files]);

  // Helper to update status for a file
  const updateStatus = (idx, updates) => {
    setStatus(prev => prev.map((s, i) => i === idx ? { ...s, ...updates } : s));
    if (updates.state && ['processing', 'cancelled', 'cancelling...'].includes(updates.state.toLowerCase())) {
      setFiles(prevFiles => prevFiles.filter((_, i) => i !== idx));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video'));
    setFiles(prev => [...prev, ...droppedFiles]);
    setStatus(prev => [...prev, ...droppedFiles.map(f => ({ name: f.name, progress: 0, state: 'Queued', jobId: null, downloadUrl: null, error: null }))]);
    setIsProcessing(false);
  };

  const handleFileInput = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(f => f.type.startsWith('video'));
    setFiles(prev => [...prev, ...selectedFiles]);
    setStatus(prev => [...prev, ...selectedFiles.map(f => ({ name: f.name, progress: 0, state: 'Queued', jobId: null, downloadUrl: null, error: null }))]);
    setIsProcessing(false);
  };

  const handleRemoveFile = (idx) => {
    setFiles(prev => prev.filter((_, i) => i !== idx));
    setStatus(prev => prev.filter((_, i) => i !== idx));
    setIsProcessing(false);
  };

  const handleStart = async () => {
    setIsProcessing(true);
    setStatus(files.map(f => ({ name: f.name, progress: 0, state: 'Queued', jobId: null, downloadUrl: null, error: null })));
    files.forEach((file, idx) => processFile(file, idx));
  };

  const processFile = async (file, idx) => {
    // Prepare form data
    const formData = new FormData();
    formData.append('video', file);
    formData.append('src_lang', srcLang);
    formData.append('tgt_lang', tgtLang);
    formData.append('voice_clone', voiceClone);
    formData.append('audio_mode', audioMode);
    formData.append('original_volume', originalVolume);
    formData.append('add_captions', addCaptions);
    formData.append('caption_font_size', captionFontSize);
    updateStatus(idx, { state: 'Uploading...' });
    try {
      const resp = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error('Failed to start processing');
      const data = await resp.json();
      const jobId = data.job_id;
      updateStatus(idx, { state: 'Processing', jobId });
      pollStatus(jobId, idx);
    } catch (err) {
      updateStatus(idx, { state: 'Error', error: err.message });
    }
  };

  const pollStatus = async (jobId, idx) => {
    let done = false;
    let consecutiveErrors = 0;
    const maxConsecutiveErrors = 5;
    
    while (!done) {
      try {
        const resp = await fetch(`${API_BASE}/status/${jobId}`);
        if (!resp.ok) {
          consecutiveErrors++;
          if (consecutiveErrors >= maxConsecutiveErrors) {
            throw new Error(`Status check failed after ${maxConsecutiveErrors} attempts`);
          }
          // Wait longer on errors and continue trying
          await new Promise(res => setTimeout(res, 5000));
          continue;
        }
        
        // Reset error counter on successful response
        consecutiveErrors = 0;
        
        const data = await resp.json();
        if (data.status === 'done') {
          updateStatus(idx, { state: 'Done', progress: 100, statusMessage: data.status_message });
          // Set download URL
          updateStatus(idx, { downloadUrl: `${API_BASE}/result/${jobId}` });
          done = true;
        } else if (data.status === 'error') {
          updateStatus(idx, { state: 'Error', error: data.error, statusMessage: data.status_message });
          done = true;
        } else {
          // Optionally update progress and status message if available
          updateStatus(idx, { state: data.status, progress: data.progress || 0, statusMessage: data.status_message });
          await new Promise(res => setTimeout(res, 3000));
        }
      } catch (err) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxConsecutiveErrors) {
          updateStatus(idx, { state: 'Error', error: err.message });
          done = true;
        } else {
          // Wait longer on errors and continue trying
          await new Promise(res => setTimeout(res, 5000));
        }
      }
    }
  };

  // Cancel job handler
  const handleCancel = async (idx, jobId) => {
    updateStatus(idx, { state: 'Cancelling...', statusMessage: 'Cancelling...' });
    try {
      const resp = await fetch(`${API_BASE}/cancel/${jobId}`, { method: 'POST' });
      if (!resp.ok) throw new Error('Failed to cancel job');
      updateStatus(idx, { state: 'Cancelled', statusMessage: 'Cancelled by user' });
    } catch (err) {
      updateStatus(idx, { state: 'Error', error: err.message });
    }
  };

  return (
    <div className="app-container">
      <h1>Video Translator UI</h1>
      <div
        className="drop-area"
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
      >
        <p>Drag and drop video files here, or <label className="file-label"><input type="file" multiple accept="video/*" onChange={handleFileInput} style={{ display: 'none' }} />browse</label></p>
      </div>
      {files.length > 0 && (
        <div className="file-list">
          <h3>Selected Videos:</h3>
          <ul>
            {files.map((file, idx) => (
              <li key={idx}>
                {file.name}
                {(!status[idx] || status[idx].state === 'Queued') && (
                  <button onClick={() => handleRemoveFile(idx)}>Remove</button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="options-panel">
        <label>Source Language:
          <select value={srcLang} onChange={e => setSrcLang(e.target.value)}>
            {LANGUAGES.map(lang => <option key={lang.code} value={lang.code}>{lang.label}</option>)}
          </select>
        </label>
        <label>Target Language:
          <select value={tgtLang} onChange={e => setTgtLang(e.target.value)}>
            {LANGUAGES.map(lang => <option key={lang.code} value={lang.code}>{lang.label}</option>)}
          </select>
        </label>
        <label>Audio Mode:
          <select value={audioMode} onChange={e => setAudioMode(e.target.value)}>
            {AUDIO_MODES.map(mode => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
          </select>
        </label>
        <label>
          <input type="checkbox" checked={voiceClone} onChange={e => setVoiceClone(e.target.checked)} /> Voice Cloning
        </label>
        <label>
          <input type="checkbox" checked={addCaptions} onChange={e => setAddCaptions(e.target.checked)} /> Add Captions (burn subtitles)
        </label>
        <label>Caption Font Size:
          <input type="number" min={12} max={64} value={captionFontSize} onChange={e => setCaptionFontSize(Number(e.target.value))} />
        </label>
        {audioMode === 'overlay' && (
          <label>Original Volume (overlay):
            <input type="range" min={0} max={1} step={0.01} value={originalVolume} onChange={e => setOriginalVolume(Number(e.target.value))} />
            {originalVolume}
          </label>
        )}
      </div>
      <button className="start-btn" onClick={handleStart} disabled={files.length === 0 || isProcessing}>Start Processing</button>
      <div className="status-panel">
        {status.map((s, idx) => (
          <div key={idx} className="status-item">
            <strong>{s.name}</strong>: {s.state} {s.progress > 0 && `(${s.progress}%)`}
            {s.statusMessage && s.progress < 100 && (
              <div style={{ marginTop: 6, color: '#1976d2', fontWeight: 500, fontSize: '1.02rem' }}>{s.statusMessage}</div>
            )}
            {s.progress > 0 && s.progress < 100 && (
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${s.progress}%` }} />
              </div>
            )}
            {/* Cancel button for in-progress jobs */}
            {s.jobId && s.state && s.state.toLowerCase() === 'processing' && (
              <button style={{ marginLeft: 12, background: '#e53935', color: '#fff', border: 'none', borderRadius: 5, padding: '0.3rem 0.9rem', cursor: 'pointer', fontWeight: 600 }}
                onClick={() => handleCancel(idx, s.jobId)}
                disabled={s.state === 'Cancelling...'}
              >Cancel</button>
            )}
            {s.downloadUrl && (
              <a href={s.downloadUrl} target="_blank" rel="noopener noreferrer" style={{ marginLeft: 12 }}>
                Download
              </a>
            )}
            {s.error && <div style={{ color: 'red' }}>Error: {s.error}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
