
import React, { useEffect, useRef, useState } from 'react';

import {
  Activity, AudioLines, Bell, Check, ChevronRight,
  ClipboardList, FileText, HeartPulse, Home, Languages,
  MessageCircle, Mic, MicOff, Plus, Send, Settings,
  ShieldCheck, Sparkles, Stethoscope, Video, VideoOff,
  Volume2, Waves, BookOpen, Search
} from 'lucide-react';

const API = 'http://127.0.0.1:8000';

async function call(path, opts = {}) {
  const response = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  });

  if (!response.ok) {
    let message = 'API request failed';
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return response.json();
}

export default function App() {
  const [page, P] = useState('Overview');
  const [apiStatus, AS] = useState('Checking');
  const [model, M] = useState(null);
  const [conversation, C] = useState(null);
  const [messages, MS] = useState([
    {
      role: 'system',
      content: 'Welcome. Confirm every message before relying on it.'
    }
  ]);

  const [patient, PT] = useState('');
  const [doctor, DT] = useState('');
  const [draft, D] = useState(null);
  const [provider, PR] = useState('');
  const [busy, B] = useState(false);
  const [notice, N] = useState('');

  const [camera, CA] = useState(false);
  const [mic, MI] = useState(false);
  const [error, E] = useState('');

  const [language, LANG] = useState('English');
  const [signs, SIGNS] = useState([]);
  const [signQuery, SQ] = useState('');

  const [recognitionStatus, RS] = useState('Waiting for camera');
  const [recognizedSign, RSign] = useState('');
  const [signConfidence, SC] = useState(0);

  const video = useRef(null);
  const stream = useRef(null);
  const recognition = useRef(null);

  // Webcam recognition references
  const frameBuffer = useRef([]);
  const predictionBusy = useRef(false);
  const lastPredictionTime = useRef(0);
  const candidateSign = useRef('');
  const candidateCount = useRef(0);
  const committedSign = useRef('');

  // Initial backend loading
  useEffect(() => {
    call('/health')
      .then(() => AS('Connected'))
      .catch(() => AS('Offline'));

    call('/api/model/status')
      .then(M)
      .catch(() => M(null));

    call('/api/learn/signs')
      .then(SIGNS)
      .catch(() => SIGNS([]));
  }, []);

  // Attach webcam stream to the video element
  useEffect(() => {
    if (video.current && stream.current) {
      video.current.srcObject = stream.current;
    }
  }, [camera, page]);

  // Stop camera and microphone when component closes
  useEffect(() => {
    return () => {
      stream.current?.getTracks().forEach(track => track.stop());
      recognition.current?.stop();
    };
  }, []);

  // Capture RGB video frames for the V2 CNN + BiLSTM + Transformer.
  // Preprocessing mirrors train_v2.py: resize proportionally and letterbox
  // into a black 64x64 canvas.
  async function processVideoFrame(frame) {
    frameBuffer.current.push(frame);

    if (frameBuffer.current.length > 16) {
      frameBuffer.current.shift();
    }

    if (frameBuffer.current.length < 16) {
      RS(`Collecting video frames: ${frameBuffer.current.length}/16`);
      return;
    }

    if (predictionBusy.current) return;

    const now = Date.now();
    if (now - lastPredictionTime.current < 1200) return;

    predictionBusy.current = true;
    lastPredictionTime.current = now;

    try {
      const result = await call('/api/sign/predict', {
        method: 'POST',
        body: JSON.stringify({
          frames: frameBuffer.current
        })
      });

      if (
        result.status === 'PREDICTED_UNCONFIRMED' &&
        result.prediction
      ) {
        const prediction = String(result.prediction).trim();

        RSign(prediction);
        SC(result.confidence || 0);
        RS('Sign detected — checking stability');

        if (candidateSign.current === prediction) {
          candidateCount.current += 1;
        } else {
          candidateSign.current = prediction;
          candidateCount.current = 1;
        }

        if (
          candidateCount.current >= 2 &&
          committedSign.current !== prediction
        ) {
          PT(previous => {
            const existingWords = previous.trim().split(/\s+/).filter(Boolean);
            const lastWord = existingWords[existingWords.length - 1];

            if (
              lastWord &&
              lastWord.toLowerCase() === prediction.toLowerCase()
            ) {
              return previous;
            }

            return previous.trim()
              ? `${previous.trim()} ${prediction}`
              : prediction;
          });

          committedSign.current = prediction;
          RS(`Added "${prediction}" to patient message`);
        } else if (committedSign.current === prediction) {
          RS(`Recognised: ${prediction}`);
        }
      } else {
        RS(result.message || 'Sign uncertain. Please repeat the gesture.');
        candidateSign.current = '';
        candidateCount.current = 0;
        committedSign.current = '';
      }
    } catch (err) {
      RS('Recognition error: ' + err.message);
    } finally {
      predictionBusy.current = false;
    }
  }

  // Capture webcam frames directly; MediaPipe is not used by V2.
  useEffect(() => {
    if (
      !camera ||
      !['Patient workspace', 'Live Room'].includes(page)
    ) {
      return;
    }

    let active = true;
    let animationId;
    let lastFrameTime = 0;
    let captureBusy = false;

    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext('2d', {
      willReadFrequently: true
    });

    frameBuffer.current = [];
    predictionBusy.current = false;
    lastPredictionTime.current = 0;
    candidateSign.current = '';
    candidateCount.current = 0;
    committedSign.current = '';

    RS('Preparing webcam frames for SignBridge V2...');

    function captureFrame() {
      if (!active) return;

      const currentVideo = video.current;
      const now = Date.now();

      if (
        currentVideo &&
        currentVideo.readyState >= 2 &&
        currentVideo.videoWidth > 0 &&
        currentVideo.videoHeight > 0 &&
        !captureBusy &&
        now - lastFrameTime >= 100
      ) {
        captureBusy = true;
        lastFrameTime = now;

        try {
          const width = currentVideo.videoWidth;
          const height = currentVideo.videoHeight;
          const scale = 64 / Math.max(width, height);
          const drawWidth = Math.max(1, Math.floor(width * scale));
          const drawHeight = Math.max(1, Math.floor(height * scale));
          const x = Math.floor((64 - drawWidth) / 2);
          const y = Math.floor((64 - drawHeight) / 2);

          context.clearRect(0, 0, 64, 64);
          context.fillStyle = 'black';
          context.fillRect(0, 0, 64, 64);
          context.drawImage(
            currentVideo,
            0, 0, width, height,
            x, y, drawWidth, drawHeight
          );

          const pixels = context.getImageData(0, 0, 64, 64).data;
          const frame = [];

          for (let row = 0; row < 64; row++) {
            const pixelRow = [];

            for (let col = 0; col < 64; col++) {
              const offset = (row * 64 + col) * 4;
              pixelRow.push([
                pixels[offset],
                pixels[offset + 1],
                pixels[offset + 2]
              ]);
            }

            frame.push(pixelRow);
          }

          processVideoFrame(frame);
        } catch (err) {
          RS('Camera frame error: ' + err.message);
        } finally {
          captureBusy = false;
        }
      }

      if (active) {
        animationId = requestAnimationFrame(captureFrame);
      }
    }

    RS('Show an emergency ISL sign to the camera');
    captureFrame();

    return () => {
      active = false;
      cancelAnimationFrame(animationId);
      frameBuffer.current = [];
      predictionBusy.current = false;
    };
  }, [camera, page]);

  async function newChat() {
    try {
      const chat = await call('/api/conversations', {
        method: 'POST',
        body: JSON.stringify({
          title: 'Hospital communication'
        })
      });

      C(chat);
      MS([]);
      P('Live Room');
    } catch (err) {
      N(err.message);
    }
  }

  async function send(content, role) {
    if (!content.trim()) return;

    try {
      let chat = conversation;

      if (!chat) {
        chat = await call('/api/conversations', {
          method: 'POST',
          body: JSON.stringify({
            title: 'Hospital communication'
          })
        });
        C(chat);
      }

      const message = await call(
        `/api/conversations/${chat.id}/messages`,
        {
          method: 'POST',
          body: JSON.stringify({
            role,
            content,
            confirmed: true,
            source: 'user-confirmed'
          })
        }
      );

      MS(previous => [...previous, message]);

      if (message.emergency_flags?.length) {
        N(
          'Staff attention flag: ' +
          message.emergency_flags.join(', ') +
          '. Not clinical triage.'
        );
      }

      if (role === 'patient') PT('');
      if (role === 'doctor') DT('');

      D(null);
    } catch (err) {
      N(err.message);
    }
  }

  // Existing GenAI integration
  async function makeDraft(isDoctor = false) {
    const value = isDoctor ? doctor : patient;

    if (!value.trim()) {
      N('Enter a message first.');
      return;
    }

    B(true);
    D(null);

    try {
      const result = await call(
        isDoctor
          ? '/api/ai/doctor-response'
          : '/api/ai/patient-message',
        {
          method: 'POST',
          body: JSON.stringify(
            isDoctor
              ? { text: value, language }
              : {
                  text: value,
                  context: messages
                    .slice(-4)
                    .map(message => message.content)
                }
          )
        }
      );

      D(result.text);
      PR(result.provider || 'AI service');

      if (result.note) {
        N(result.note);
      } else {
        N('AI draft generated. Review before sending.');
      }
    } catch (err) {
      N('AI improvement failed: ' + err.message);
    } finally {
      B(false);
    }
  }

  async function cam() {
    if (camera) {
      stream.current?.getTracks().forEach(track => track.stop());
      stream.current = null;
      CA(false);
      RS('Camera stopped');
      RSign('');
      SC(0);
      frameBuffer.current = [];
      return;
    }

    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      CA(true);
      E('');
      RS('Starting camera...');
    } catch (err) {
      E('Camera permission denied or camera unavailable.');
    }
  }

  function dictate() {
    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      N('Speech recognition is unavailable in this browser.');
      return;
    }

    if (mic) {
      recognition.current?.stop();
      MI(false);
      return;
    }

    const recognizer = new SpeechRecognition();

    recognizer.lang =
      language === 'Kannada'
        ? 'kn-IN'
        : language === 'Hindi'
          ? 'hi-IN'
          : 'en-IN';

    recognizer.interimResults = true;

    recognizer.onresult = event => {
      let transcript = '';

      for (
        let i = event.resultIndex;
        i < event.results.length;
        i++
      ) {
        transcript += event.results[i][0].transcript;
      }

      DT(transcript);
    };

    recognizer.onerror = event => {
      N('Speech recognition: ' + event.error);
    };

    recognizer.onend = () => MI(false);

    recognition.current = recognizer;
    recognizer.start();
    MI(true);
  }

  function speak(text) {
    if (!window.speechSynthesis) {
      N('Text-to-speech unavailable');
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang =
      language === 'Kannada'
        ? 'kn-IN'
        : language === 'Hindi'
          ? 'hi-IN'
          : 'en-IN';

    window.speechSynthesis.speak(utterance);
  }

  const nav = [
    ['Overview', Home],
    ['Patient workspace', HeartPulse],
    ['Doctor workspace', Stethoscope],
    ['Live Room', MessageCircle],
    ['Learn ISL', BookOpen],
    ['Reports', ClipboardList],
    ['Settings', Settings]
  ];

  return (
    <div className="app">
      <aside>
        <div className="brand">
          <div className="mark"><Waves /></div>
          <div>
            <b>signbridge<span>AI</span></b>
            <small>CARE WITHOUT BARRIERS</small>
          </div>
        </div>

        <label>WORKSPACE</label>

        {nav.map(([name, Icon]) => (
          <button
            className={page === name ? 'selected' : ''}
            onClick={() => P(name)}
            key={name}
          >
            <Icon size={17} />
            {name}
          </button>
        ))}

        <div className="side-foot">
          <ShieldCheck />
          <b>Private local session</b>
          <small>Development workspace</small>
        </div>
      </aside>

      <main>
        <header>
          <span>
            Workspace <ChevronRight size={13} /> <b>{page}</b>
          </span>

          <div>
            <i className={apiStatus === 'Connected' ? 'good' : ''} />
            {apiStatus}
            <button onClick={() => N('No new notifications')}>
              <Bell size={17} />
            </button>
            <b className="avatar">SB</b>
          </div>
        </header>

        <article>
          {notice && (
            <div className="notice">
              {notice}
              <button onClick={() => N('')}>×</button>
            </div>
          )}

          {page === 'Overview' && (
            <>
              <section className="hero">
                <div>
                  <label>ACCESSIBLE CARE, CONNECTED</label>
                  <h1>
                    Communication that
                    <br />
                    <em>understands every voice.</em>
                  </h1>
                  <p>
                    A shared space for patients and care teams to
                    communicate clearly, confidently and with dignity.
                  </p>
                  <button className="primary" onClick={newChat}>
                    <MessageCircle size={16} />
                    Start a conversation
                  </button>
                </div>

                <div className="orb">
                  <HeartPulse size={54} />
                  <b>Care in<br />every sign.</b>
                  <span>Human-centred care</span>
                </div>
              </section>

              <h2>Choose how to connect</h2>

              <div className="cards">
                {[
                  [
                    'Patient workspace',
                    'I am a patient',
                    'Share needs through signing, text or communication prompts.',
                    HeartPulse
                  ],
                  [
                    'Doctor workspace',
                    'I am a clinician',
                    'Receive patient messages and respond through speech or text.',
                    Stethoscope
                  ],
                  [
                    'Live Room',
                    'Shared communication',
                    'Bring both sides together in one accessible conversation.',
                    AudioLines
                  ]
                ].map(([target, title, description, Icon]) => (
                  <button
                    className="card"
                    onClick={() => P(target)}
                    key={target}
                  >
                    <Icon />
                    <small>ACCESSIBLE WORKSPACE</small>
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <b>
                      Open workspace <ChevronRight size={15} />
                    </b>
                  </button>
                ))}
              </div>

              <div className="safety">
                <ShieldCheck />
                <div>
                  <b>Human-confirmed communication</b>
                  <p>
                    AI suggestions are drafts. Patients and clinicians
                    stay in control.
                  </p>
                </div>
                <Sparkles />
                <div>
                  <b>AI with clear boundaries</b>
                  <p>
                    Language assistance only — never diagnosis or
                    prescription.
                  </p>
                </div>
              </div>
            </>
          )}

          {page === 'Patient workspace' && (
            <>
              <Title
                title="Your voice matters."
                sub="Sign in front of the camera. Recognised signs will be added to your message."
              />

              <div className="cols">
                <section className="panel">
                  <Head
                    title="Live sign recognition"
                    sub="Webcam · CNN + BiLSTM + Transformer"
                  />

                  <div className="camera">
                    {camera ? (
                      <video
                        ref={video}
                        autoPlay
                        muted
                        playsInline
                      />
                    ) : (
                      <div>
                        <Video />
                        <b>Camera preview</b>
                        <small>Enable the camera to start recognition.</small>
                      </div>
                    )}
                  </div>

                  <button className="secondary wide" onClick={cam}>
                    {camera ? <VideoOff /> : <Video />}
                    {camera ? 'Stop camera' : 'Start recognition'}
                  </button>

                  {error && <p className="error">{error}</p>}

                  <div className="warning">
                    <Activity />
                    <div>
                      <b>{recognitionStatus}</b>
                      <small>
                        {model?.video?.ready
                          ? 'Emergency V2 model loaded'
                          : model?.ready
                            ? 'Trained model loaded'
                            : 'Checking emergency V2 model'}
                      </small>
                    </div>
                  </div>

                  {recognizedSign && (
                    <div className="info">
                      <b>Latest recognised sign</b>
                      <span>{recognizedSign}</span>
                      <small>
                        Confidence: {(signConfidence * 100).toFixed(1)}%
                      </small>
                    </div>
                  )}
                </section>

                <section className="panel">
                  <Head
                    title="Patient message"
                    sub="Recognised signs appear here automatically. You can also type."
                  />

                  <textarea
                    value={patient}
                    onChange={event => PT(event.target.value)}
                    rows="5"
                    placeholder="Your recognised signs will appear here..."
                  />

                  <button
                    className="secondary wide"
                    disabled={busy || !patient.trim()}
                    onClick={() => makeDraft(false)}
                  >
                    <Sparkles />
                    {busy ? 'Preparing AI draft...' : 'Improve with AI'}
                  </button>

                  {draft && (
                    <Draft
                      text={draft}
                      provider={provider}
                      onSend={() => send(draft, 'patient')}
                      onDiscard={() => D(null)}
                    />
                  )}

                  <h4>COMMON NEEDS</h4>

                  <div className="quick">
                    {[
                      'I need help',
                      'I am in pain',
                      'Please call my family',
                      'I need water',
                      'I feel dizzy',
                      'I need a doctor'
                    ].map(item => (
                      <button
                        onClick={() => PT(item)}
                        key={item}
                      >
                        {item}
                        <Plus size={13} />
                      </button>
                    ))}
                  </div>
                </section>
              </div>
            </>
          )}

          {page === 'Doctor workspace' && (
            <>
              <Title
                title="Listen. Understand. Respond."
                sub="Prepare a clear and accessible response for the patient."
              />

              <div className="cols">
                <section className="panel">
                  <Head
                    title="Compose a response"
                    sub="Type a reply or dictate using your microphone"
                  />

                  <label className="field-label">
                    Response language
                  </label>

                  <select
                    value={language}
                    onChange={event => LANG(event.target.value)}
                  >
                    <option>English</option>
                    <option>Kannada</option>
                    <option>Hindi</option>
                  </select>

                  <button className="secondary" onClick={dictate}>
                    {mic ? <MicOff /> : <Mic />}
                    {mic ? 'Stop listening' : 'Dictate response'}
                  </button>

                  <textarea
                    value={doctor}
                    onChange={event => DT(event.target.value)}
                    rows="6"
                    placeholder="Doctor response..."
                  />

                  <div className="actions">
                    <button
                      className="secondary"
                      disabled={!doctor.trim() || busy}
                      onClick={() => makeDraft(true)}
                    >
                      <Sparkles /> Simplify language
                    </button>

                    <button
                      className="primary"
                      onClick={() => send(draft || doctor, 'doctor')}
                      disabled={!doctor.trim() && !draft}
                    >
                      <Send /> Send to patient
                    </button>

                    <button
                      className="secondary"
                      onClick={() => speak(draft || doctor)}
                    >
                      <Volume2 /> Read aloud
                    </button>
                  </div>

                  {draft && (
                    <Draft
                      text={draft}
                      provider={provider}
                      onSend={() => send(draft, 'doctor')}
                      onDiscard={() => D(null)}
                    />
                  )}
                </section>

                <section className="panel guidance">
                  <ShieldCheck />
                  <h3>Communication with care</h3>
                  <p>
                    AI can help simplify language, but it cannot replace
                    clinical judgement or a qualified interpreter.
                  </p>
                  <ul>
                    <li>Review generated explanations.</li>
                    <li>Preserve medical meaning and uncertainty.</li>
                    <li>Confirm patient understanding.</li>
                    <li>Use a qualified ISL interpreter when needed.</li>
                  </ul>
                </section>
              </div>
            </>
          )}

          {page === 'Live Room' && (
            <>
              <Title
                title="Live communication room"
                sub="Shared communication space. Messages are stored in the local development database."
              />

              <div className="roombar">
                <b>
                  {conversation
                    ? 'Conversation #' + conversation.id
                    : 'No active conversation'}
                </b>
                <button className="secondary" onClick={newChat}>
                  <Plus /> New session
                </button>
              </div>

              <div className="cols">
                <section className="panel">
                  <Head
                    title="Patient side"
                    sub="Live signing and recognised message"
                  />

                  <div className="camera mini">
                    {camera ? (
                      <video
                        ref={video}
                        autoPlay
                        muted
                        playsInline
                      />
                    ) : (
                      <div>
                        <Video />
                        <small>Camera preview off</small>
                      </div>
                    )}
                  </div>

                  <button className="secondary wide" onClick={cam}>
                    {camera ? 'Stop camera' : 'Enable camera'}
                  </button>

                  <p className="muted">{recognitionStatus}</p>

                  <textarea
                    value={patient}
                    onChange={event => PT(event.target.value)}
                    rows="3"
                    placeholder="Patient message..."
                  />

                  <button
                    className="secondary wide"
                    onClick={() => makeDraft(false)}
                    disabled={!patient.trim() || busy}
                  >
                    <Sparkles /> Improve with AI
                  </button>

                  {draft && (
                    <Draft
                      text={draft}
                      provider={provider}
                      onSend={() => send(draft, 'patient')}
                      onDiscard={() => D(null)}
                    />
                  )}

                  <button
                    className="primary wide"
                    onClick={() => send(patient, 'patient')}
                    disabled={!patient.trim()}
                  >
                    <Send /> Confirm & send
                  </button>
                </section>

                <section className="panel">
                  <Head
                    title="Doctor side"
                    sub="Speech, text and accessible replies"
                  />

                  <textarea
                    value={doctor}
                    onChange={event => DT(event.target.value)}
                    rows="3"
                    placeholder="Doctor response..."
                  />

                  <div className="actions">
                    <button className="secondary" onClick={dictate}>
                      {mic ? <MicOff /> : <Mic />} Dictate
                    </button>

                    <button
                      className="secondary"
                      onClick={() => speak(doctor)}
                    >
                      <Volume2 /> Read aloud
                    </button>
                  </div>

                  <button
                    className="primary wide"
                    onClick={() => send(doctor, 'doctor')}
                    disabled={!doctor.trim()}
                  >
                    <Send /> Send to patient
                  </button>
                </section>
              </div>

              <section className="panel">
                <Head
                  title="Conversation timeline"
                  sub="Human-confirmed entries"
                />

                {messages.length ? (
                  messages.map((message, index) => (
                    <div className="msg" key={message.id || index}>
                      <b>{message.role}</b>
                      <p>{message.content}</p>
                      <small>
                        {message.confirmed
                          ? 'Confirmed'
                          : 'Review required'}
                      </small>
                    </div>
                  ))
                ) : (
                  <p className="muted">
                    No messages yet. Start by sending a confirmed message.
                  </p>
                )}
              </section>
            </>
          )}

          {page === 'Learn ISL' && (
            <>
              <Title
                title="Learn Indian Sign Language"
                sub="Explore the vocabulary included in the current SignBridge datasets. Meanings are gloss explanations; use the dataset's original sign examples to learn the actual hand movement."
              />

              <section className="panel">
                <div className="searchbar">
                  <Search />
                  <input
                    value={signQuery}
                    onChange={event => SQ(event.target.value)}
                    placeholder="Search signs, e.g. pain, doctor, help..."
                  />
                </div>

                <div className="cards">
                  {signs
                    .filter(sign =>
                      (
                        sign.sign +
                        ' ' +
                        sign.category +
                        ' ' +
                        sign.meaning
                      )
                        .toLowerCase()
                        .includes(signQuery.toLowerCase())
                    )
                    .map(sign => (
                      <div className="card sign-card" key={sign.sign}>
                        <small>{sign.category}</small>
                        <h3>{sign.sign}</h3>
                        <p>{sign.meaning}</p>
                        <small>Source: {sign.dataset}</small>
                      </div>
                    ))}
                </div>

                <p className="muted">
                  The dictionary currently explains 26 dataset labels.
                  Do not use the written gloss as a substitute for a
                  verified sign demonstration.
                </p>
              </section>
            </>
          )}

          {page === 'Reports' && (
            <>
              <Title
                title="Conversation reports"
                sub="Communication summaries for the current local session."
              />

              <section className="panel">
                <Head
                  title="Conversation summary"
                  sub="Select a conversation from the Live Room to review its timeline"
                />

                <p className="muted">
                  {conversation
                    ? 'Conversation #' + conversation.id
                    : 'No conversation selected.'}
                </p>

                {messages.map((message, index) => (
                  <div className="msg" key={index}>
                    <b>{message.role}</b>
                    <p>{message.content}</p>
                  </div>
                ))}

                <p className="muted">
                  Communication summary only; not a medical record or diagnosis.
                </p>
              </section>
            </>
          )}

          {page === 'Settings' && (
            <>
              <Title
                title="Workspace settings"
                sub="Local development configuration and accessibility status."
              />

              <div className="cols">
                <section className="panel">
                  <Head title="Language & accessibility" />
                  <Info
                    label="Doctor response language"
                    value="English / Kannada / Hindi"
                  />
                  <Info
                    label="Speech recognition"
                    value="Browser · en-IN"
                  />
                  <Info
                    label="Text-to-speech"
                    value="Browser voices"
                  />
                </section>

                <section className="panel">
                  <Head title="AI & recognition status" />
                  <Info label="Backend API" value={apiStatus} />
                  <Info
                    label="Video model"
                    value={
                      model?.video?.ready || model?.ready
                        ? 'Checkpoint loaded'
                        : 'Not loaded'
                    }
                  />
                  <Info
                    label="Generative AI"
                    value="Groq · GenAI"
                  />
                </section>
              </div>
            </>
          )}

          <footer>
            SignBridge AI · Assistive communication prototype
            <span>
              <ShieldCheck size={13} />
              Human confirmation required
            </span>
          </footer>
        </article>
      </main>
    </div>
  );
}

function Title({ title, sub }) {
  return (
    <div className="title">
      <label>SHARED ACCESSIBLE CARE</label>
      <h1>{title}</h1>
      <p>{sub}</p>
    </div>
  );
}

function Head({ title, sub }) {
  return (
    <div className="head">
      <div>
        <h3>{title}</h3>
        <small>{sub}</small>
      </div>
      <i />
    </div>
  );
}

function Draft({ text, provider, onSend, onDiscard }) {
  return (
    <div className="draft">
      <small>REVIEW DRAFT · {provider}</small>
      <p>{text}</p>
      <button className="primary" onClick={onSend}>
        <Check /> Confirm & send
      </button>
      <button className="link" onClick={onDiscard}>
        Discard
      </button>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info">
      <b>{label}</b>
      <span>{value}</span>
    </div>
  );
}
