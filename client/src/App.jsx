import React, { useState, useRef, useEffect } from 'react';
import { Camera, RefreshCw, Layers, ShieldAlert, Cpu, Award, User, Clock, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [cameraId, setCameraId] = useState('cam_01');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analyticsResult, setAnalyticsResult] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [activeTab, setActiveTab] = useState('canvas'); // 'canvas' or 'json'

  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Redraw canvas when image or analytics result changes
  useEffect(() => {
    if (!selectedFile) return;

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setImageSize({ width: img.width, height: img.height });
      drawCanvas(img, analyticsResult?.faces || []);
    };
    img.src = URL.createObjectURL(selectedFile);

    // Clean up object URL on unmount or file change
    return () => URL.revokeObjectURL(img.src);
  }, [selectedFile, analyticsResult]);

  const drawCanvas = (image, faces) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions to match the source image exactly
    canvas.width = image.width;
    canvas.height = image.height;
    
    // Draw original image as the background
    ctx.drawImage(image, 0, 0);

    // Overlay faces bounding boxes and labels
    faces.forEach((face) => {
      const [x1, y1, x2, y2] = face.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      // Color coding based on estimated gender
      const isMale = face.gender === 'male';
      const themeColor = isMale ? '#3b82f6' : '#ec4899'; // Blue vs Pink

      // 1. Draw Bounding Box border
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = Math.max(3, Math.round(image.width * 0.004)); // Responsive line width
      ctx.strokeRect(x1, y1, width, height);

      // 2. Draw Corner Accents for a futuristic HUD feel
      const accentLen = Math.min(width, height) * 0.2;
      ctx.fillStyle = themeColor;
      
      // Top-Left corner
      ctx.fillRect(x1 - ctx.lineWidth/2, y1 - ctx.lineWidth/2, accentLen, ctx.lineWidth * 2);
      ctx.fillRect(x1 - ctx.lineWidth/2, y1 - ctx.lineWidth/2, ctx.lineWidth * 2, accentLen);
      // Top-Right corner
      ctx.fillRect(x2 - accentLen + ctx.lineWidth/2, y1 - ctx.lineWidth/2, accentLen, ctx.lineWidth * 2);
      ctx.fillRect(x2 - ctx.lineWidth * 1.5, y1 - ctx.lineWidth/2, ctx.lineWidth * 2, accentLen);
      // Bottom-Left corner
      ctx.fillRect(x1 - ctx.lineWidth/2, y2 - ctx.lineWidth * 1.5, accentLen, ctx.lineWidth * 2);
      ctx.fillRect(x1 - ctx.lineWidth/2, y2 - accentLen + ctx.lineWidth/2, ctx.lineWidth * 2, accentLen);
      // Bottom-Right corner
      ctx.fillRect(x2 - accentLen + ctx.lineWidth/2, y2 - ctx.lineWidth * 1.5, accentLen, ctx.lineWidth * 2);
      ctx.fillRect(x2 - ctx.lineWidth * 1.5, y2 - accentLen + ctx.lineWidth/2, ctx.lineWidth * 2, accentLen);

      // 3. Draw Label Banner on top of bounding box
      const labelText = `ID ${face.track_id} | ${face.gender === 'male' ? 'M' : 'F'} | ${face.age} años (${face.age_range})`;
      
      // Configure responsive text size
      const fontSize = Math.max(14, Math.round(image.width * 0.018));
      ctx.font = `bold ${fontSize}px sans-serif`;
      
      const textWidth = ctx.measureText(labelText).width;
      const labelPadding = 8;
      const labelHeight = fontSize + labelPadding * 2;

      // Draw semi-transparent background fill for label
      ctx.fillStyle = `${themeColor}cc`; // Add transparency (cc = 80%)
      ctx.fillRect(x1 - ctx.lineWidth/2, y1 - labelHeight, textWidth + labelPadding * 2, labelHeight);

      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x1 + labelPadding, y1 - labelPadding);
    });
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setAnalyticsResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Por favor selecciona una imagen primero.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('camera_id', cameraId);
    formData.append('timestamp', Math.floor(Date.now() / 1000));
    formData.append('image', selectedFile);

    // In Docker-compose, Nginx proxies API on the same domain.
    // Use relative path by default, can override with VITE_API_URL for local dev.
    const apiUrl ='/api/v1/analyze';

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Servidor sobrecargado (Límite de Backpressure alcanzado). Intenta de nuevo en unos segundos.');
        }
        const errData = await response.json();
        throw new Error(errData.detail || 'Fallo al procesar el análisis de la imagen.');
      }

      const data = await response.json();
      setAnalyticsResult(data);
    } catch (err) {
      setError(err.message || 'Error de red al conectar con el servidor.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-8 pb-6 border-b border-slate-800">
        <div className="flex items-center space-x-3 mb-4 md:mb-0">
          <div className="p-2.5 bg-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Cpu className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Demographics Engine</h1>
            <p className="text-xs text-slate-400 font-mono">Edge Face Detection, Tracking & Demographics</p>
          </div>
        </div>
        <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2 text-xs font-mono text-slate-400">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Síncrono (Baja Latencia)</span>
        </div>
      </header>

      {/* ERROR BANNER */}
      {error && (
        <div className="mb-6 p-4 bg-red-950/40 border border-red-800 text-red-200 rounded-xl flex items-start space-x-3">
          <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-sm">Error en la petición</p>
            <p className="text-xs mt-0.5 opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* CORE INTERFACE */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* PANEL IZQUIERDO: CONTROLES */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
            <h2 className="text-base font-bold text-white mb-4 flex items-center space-x-2">
              <Camera className="w-5 h-5 text-blue-500" />
              <span>Configuración de Frame</span>
            </h2>

            <form onSubmit={handleAnalyze} className="space-y-5">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Cámara Source (ID)
                </label>
                <input
                  type="text"
                  value={cameraId}
                  onChange={(e) => setCameraId(e.target.value)}
                  placeholder="e.g. cam_01"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 font-mono"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Cargar Imagen
                </label>
                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-800 border-dashed rounded-xl hover:border-slate-700 transition cursor-pointer relative bg-slate-950">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="space-y-1 text-center">
                    <Layers className="mx-auto h-10 w-10 text-slate-500" />
                    <div className="flex text-sm text-slate-400">
                      <span className="relative rounded-md font-semibold text-blue-500 hover:text-blue-400">
                        Sube un archivo
                      </span>
                      <p className="pl-1">o arrastra y suelta</p>
                    </div>
                    <p className="text-xs text-slate-500">PNG, JPG, WEBP hasta 10MB</p>
                  </div>
                </div>
                {selectedFile && (
                  <p className="text-xs font-mono text-emerald-400 mt-2 flex items-center">
                    <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                    <span>{selectedFile.name}</span>
                  </p>
                )}
              </div>

              <button
                type="submit"
                disabled={loading || !selectedFile}
                className={`w-full flex items-center justify-center space-x-2 py-3.5 px-4 rounded-xl text-sm font-semibold transition shadow-lg ${
                  loading || !selectedFile
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed shadow-none'
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-500/10'
                }`}
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Analizando con IA...</span>
                  </>
                ) : (
                  <span>Ejecutar Pipeline Síncrono</span>
                )}
              </button>
            </form>
          </div>

          {/* PANEL DE MÉTRICAS */}
          {analyticsResult && (
            <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                Performance Telemetry
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/60">
                  <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
                    <Clock className="w-3.5 h-3.5 text-blue-500" />
                    <span>Inferencia</span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-white">
                    {analyticsResult.processing_time_ms}
                    <span className="text-xs text-slate-400 font-normal ml-1">ms</span>
                  </div>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/60">
                  <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
                    <User className="w-3.5 h-3.5 text-pink-500" />
                    <span>Rostros</span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-white">
                    {analyticsResult.faces.length}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* PANEL DERECHO: VISUALIZADOR */}
        <div className="lg:col-span-8 flex flex-col h-full space-y-6">
          <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 flex flex-col flex-grow shadow-xl min-h-[450px]">
            {/* TABS CONTROLES */}
            <div className="flex justify-between items-center border-b border-slate-800/80 pb-3 mb-4">
              <div className="flex space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setActiveTab('canvas')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                    activeTab === 'canvas' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Visualizador Canvas
                </button>
                <button
                  onClick={() => setActiveTab('json')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                    activeTab === 'json' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  JSON Response
                </button>
              </div>
              {imageSize.width > 0 && (
                <div className="text-xs font-mono text-slate-500">
                  Res: {imageSize.width}x{imageSize.height}px
                </div>
              )}
            </div>

            {/* TAB CONTENT: CANVAS */}
            {activeTab === 'canvas' && (
              <div className="flex-grow flex items-center justify-center bg-slate-950 rounded-xl overflow-hidden border border-slate-800 p-2 relative">
                {!selectedFile ? (
                  <div className="text-center py-20 text-slate-500 space-y-2">
                    <Layers className="w-12 h-12 mx-auto text-slate-600 opacity-60 animate-pulse" />
                    <p className="text-sm">Ninguna imagen seleccionada</p>
                    <p className="text-xs max-w-xs mx-auto opacity-70">Sube una foto en el panel izquierdo para procesar la estimación facial</p>
                  </div>
                ) : (
                  <div className="max-w-full max-h-[600px] overflow-auto">
                    <canvas 
                      ref={canvasRef} 
                      className="max-w-full h-auto block rounded-lg shadow-2xl border border-slate-800/60"
                    />
                  </div>
                )}
              </div>
            )}

            {/* TAB CONTENT: JSON */}
            {activeTab === 'json' && (
              <div className="flex-grow bg-slate-950 rounded-xl p-4 overflow-auto border border-slate-800 font-mono text-xs text-emerald-400 h-[450px]">
                {analyticsResult ? (
                  <pre>{JSON.stringify(analyticsResult, null, 2)}</pre>
                ) : (
                  <p className="text-slate-500 text-center py-20">No hay datos disponibles. Ejecuta el pipeline para ver el JSON.</p>
                )}
              </div>
            )}
          </div>

          {/* TABLA DE ROSTROS DETECTADOS */}
          {analyticsResult && analyticsResult.faces.length > 0 && (
            <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
              <h3 className="text-sm font-bold mb-4 uppercase tracking-wider text-slate-400 flex items-center space-x-2">
                <Award className="w-5 h-5 text-yellow-500" />
                <span>Rostros Consolidados (Agregación Temporal)</span>
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase font-semibold font-mono tracking-wider">
                      <th className="py-3 px-4">Track ID</th>
                      <th className="py-3 px-4">Género</th>
                      <th className="py-3 px-4">Conf. Género</th>
                      <th className="py-3 px-4">Edad Promedio</th>
                      <th className="py-3 px-4">Rango</th>
                      <th className="py-3 px-4">Conf. Tracking</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {analyticsResult.faces.map((face) => (
                      <tr key={face.track_id} className="hover:bg-slate-950/40 transition">
                        <td className="py-3.5 px-4 font-bold text-white">#{face.track_id}</td>
                        <td className="py-3.5 px-4">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-md font-semibold ${
                            face.gender === 'male' ? 'bg-blue-950 text-blue-400 border border-blue-900/50' : 'bg-pink-950 text-pink-400 border border-pink-900/50'
                          }`}>
                            {face.gender === 'male' ? 'MALE' : 'FEMALE'}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-slate-300">{(face.gender_confidence * 100).toFixed(0)}%</td>
                        <td className="py-3.5 px-4 text-white font-bold">{face.age} años</td>
                        <td className="py-3.5 px-4 text-slate-400">{face.age_range}</td>
                        <td className="py-3.5 px-4 text-slate-300">{face.confidence.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
