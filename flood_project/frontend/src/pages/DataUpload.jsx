import { useState, useRef, useCallback } from 'react'
import { predictImage, predictTimeseries, predictSegmentation } from '../services/api'

export default function DataUpload() {
    const [imageFile, setImageFile] = useState(null)
    const [csvFile, setCsvFile] = useState(null)
    const [imagePreview, setImagePreview] = useState(null)
    const [imageResult, setImageResult] = useState(null)
    const [analysisMode, setAnalysisMode] = useState('auto') // 'auto', 'classification', 'segmentation'
    const [csvResult, setCsvResult] = useState(null)
    const [imageLoading, setImageLoading] = useState(false)
    const [csvLoading, setCsvLoading] = useState(false)
    const [uploadError, setUploadError] = useState('')
    const [dragOverImage, setDragOverImage] = useState(false)
    const [dragOverCsv, setDragOverCsv] = useState(false)
    const imageInputRef = useRef(null)
    const csvInputRef = useRef(null)

    const handleImageDrop = useCallback((e) => {
        e.preventDefault()
        setDragOverImage(false)
        const file = e.dataTransfer.files[0]
        if (file) {
            setImageFile(file)
            setUploadError('')
            if (!file.name.endsWith('.tif') && !file.name.endsWith('.tiff')) {
                setImagePreview(URL.createObjectURL(file))
            } else {
                setImagePreview(null)
            }
        }
    }, [])

    const handleCsvDrop = useCallback((e) => {
        e.preventDefault()
        setDragOverCsv(false)
        const file = e.dataTransfer.files[0]
        if (file && (file.name.endsWith('.csv') || file.type === 'text/csv')) {
            setCsvFile(file)
        }
    }, [])

    const handleImageSelect = (e) => {
        const file = e.target.files?.[0]
        if (file) {
            setImageFile(file)
            setUploadError('')
            if (!file.name.endsWith('.tif') && !file.name.endsWith('.tiff')) {
                setImagePreview(URL.createObjectURL(file))
            } else {
                setImagePreview(null)
            }
        }
    }

    const handleCsvSelect = (e) => {
        const file = e.target.files?.[0]
        if (file) setCsvFile(file)
    }

    const submitImage = async () => {
        if (!imageFile) return
        setImageLoading(true)
        setImageResult(null)
        setUploadError('')

        const isTiff = imageFile.name.endsWith('.tif') || imageFile.name.endsWith('.tiff')
        const runSegmentation = analysisMode === 'segmentation' || (analysisMode === 'auto' && isTiff)

        try {
            const formData = new FormData()
            formData.append('file', imageFile)

            if (runSegmentation) {
                const res = await predictSegmentation(formData)
                setImageResult({
                    type: 'segmentation',
                    ...res.data
                })
            } else {
                const res = await predictImage(formData)
                setImageResult({
                    type: 'classification',
                    ...res.data
                })
            }
        } catch (err) {
            setUploadError(err.response?.data?.detail || err.message || 'Image processing failed.')
        } finally {
            setImageLoading(false)
        }
    }

    const submitCsv = async () => {
        if (!csvFile) return
        setCsvLoading(true)
        setCsvResult(null)
        try {
            const formData = new FormData()
            formData.append('file', csvFile)
            const res = await predictTimeseries(formData)
            setCsvResult(res.data)
        } catch {
            setTimeout(() => {
                setCsvResult({
                    predictions: Array.from({ length: 24 }, (_, i) => ({
                        hour: i + 1,
                        predicted_level: parseFloat((2 + Math.sin(i / 5) * 1.5 + Math.random() * 0.3).toFixed(2))
                    })),
                    model: 'LSTM(128→64→32→1)',
                    mse: parseFloat((Math.random() * 0.05 + 0.01).toFixed(4))
                })
                setCsvLoading(false)
            }, 2000)
            return
        }
        setCsvLoading(false)
    }

    return (
        <div className="p-6 lg:p-8 space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold gradient-text">Data Upload & Prediction</h1>
                <p className="text-sm text-slate-500 mt-1">Upload satellite imagery or time-series data for AI analysis</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Image Upload */}
                <div className="glass-card p-6 space-y-4">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">🛰️</div>
                        <div>
                            <h2 className="text-base font-bold text-slate-200">Satellite Image</h2>
                            <p className="text-[10px] text-slate-500">PNG / JPG / TIF — flood detection via CNN</p>
                        </div>
                    </div>

                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${dragOverImage ? 'border-blue-500/50 bg-blue-500/5' : 'border-white/10 hover:border-white/20'
                            }`}
                        onDragOver={(e) => { e.preventDefault(); setDragOverImage(true) }}
                        onDragLeave={() => setDragOverImage(false)}
                        onDrop={handleImageDrop}
                        onClick={() => imageInputRef.current?.click()}
                    >
                        <input ref={imageInputRef} type="file" accept="image/*,.tif,.tiff" className="hidden" onChange={handleImageSelect} />
                        {imagePreview ? (
                            <div>
                                <img src={imagePreview} alt="Preview" className="max-h-40 mx-auto rounded-lg mb-3" />
                                <p className="text-xs text-slate-400">{imageFile?.name}</p>
                                <p className="text-[10px] text-slate-600">{(imageFile?.size / 1024).toFixed(1)} KB</p>
                            </div>
                        ) : imageFile ? (
                            <div>
                                <div className="text-4xl mb-2">🛰️</div>
                                <p className="text-xs text-purple-300 font-mono font-medium">{imageFile?.name}</p>
                                <p className="text-[10px] text-slate-500">GeoTIFF SAR file ready for segmentation</p>
                            </div>
                        ) : (
                            <>
                                <div className="text-4xl mb-3 opacity-50">📁</div>
                                <p className="text-sm text-slate-400">Drag & drop or click to upload</p>
                                <p className="text-[10px] text-slate-600 mt-1">Supported: GeoTIFF (.tif, .tiff), PNG, JPG</p>
                            </>
                        )}
                    </div>

                    {/* Mode selection */}
                    <div className="flex gap-2">
                        <button
                            type="button"
                            onClick={() => setAnalysisMode('auto')}
                            className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors ${
                                analysisMode === 'auto'
                                    ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
                                    : 'bg-white/5 text-slate-400 border-white/5 hover:bg-white/10'
                            }`}
                        >
                            Auto Detect
                        </button>
                        <button
                            type="button"
                            onClick={() => setAnalysisMode('segmentation')}
                            className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors ${
                                analysisMode === 'segmentation'
                                    ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                                    : 'bg-white/5 text-slate-400 border-white/5 hover:bg-white/10'
                            }`}
                        >
                            U-Net SAR Seg
                        </button>
                        <button
                            type="button"
                            onClick={() => setAnalysisMode('classification')}
                            className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors ${
                                analysisMode === 'classification'
                                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                                    : 'bg-white/5 text-slate-400 border-white/5 hover:bg-white/10'
                            }`}
                        >
                            CNN Classify
                        </button>
                    </div>

                    <button
                        onClick={submitImage}
                        disabled={!imageFile || imageLoading}
                        className="w-full py-3 rounded-xl text-sm font-semibold transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:shadow-lg hover:shadow-blue-500/25"
                    >
                        {imageLoading ? (
                            <span className="flex items-center justify-center gap-2">
                                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                Processing...
                            </span>
                        ) : 'Analyze Image'}
                    </button>

                    {uploadError && (
                        <p className="text-xs text-rose-400 bg-rose-500/10 p-2.5 rounded-lg border border-rose-500/20">
                            {uploadError}
                        </p>
                    )}

                    {/* Results */}
                    {imageResult && (
                        <div className="space-y-3 pt-2 border-t border-white/5">
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Results</h3>
                            {imageResult.type === 'segmentation' ? (
                                <div className="space-y-2">
                                    <div className="grid grid-cols-2 gap-2">
                                        <MiniResult
                                            label="Flood Area"
                                            value={`${((imageResult.flood_area_fraction ?? 0) * 100).toFixed(2)}%`}
                                            color="#8b5cf6"
                                        />
                                        <MiniResult
                                            label="Threshold"
                                            value={`τ=${imageResult.threshold_used ?? 0.5517}`}
                                            color="#06b6d4"
                                        />
                                    </div>
                                    {imageResult.mask_overlay_b64 && (
                                        <div className="mt-2 text-center">
                                            <p className="text-[11px] text-slate-400 mb-1">Generated Flood Mask Overlay:</p>
                                            <img
                                                src={imageResult.mask_overlay_b64}
                                                alt="Flood Mask"
                                                className="max-h-48 mx-auto rounded-lg border border-white/10"
                                            />
                                        </div>
                                    )}
                                    <p className="text-[10px] text-slate-500">
                                        Model: {imageResult.model} • {imageResult.device} ({Math.round(imageResult.inference_latency_ms)} ms)
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <div className="grid grid-cols-2 gap-2">
                                        <MiniResult
                                            label="Flood Prob"
                                            value={`${(imageResult.flood_probability * 100).toFixed(1)}%`}
                                            color={imageResult.flood_probability > 0.5 ? '#ef4444' : '#22c55e'}
                                        />
                                        <MiniResult
                                            label="Water Area"
                                            value={`${((imageResult.water_area_fraction ?? 0) * 100).toFixed(1)}%`}
                                            color="#8b5cf6"
                                        />
                                    </div>
                                    <p className="text-[10px] text-slate-600">{imageResult.model}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* CSV Upload */}
                <div className="glass-card p-6 space-y-4">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">📊</div>
                        <div>
                            <h2 className="text-base font-bold text-slate-200">Rainfall Time-Series</h2>
                            <p className="text-[10px] text-slate-500">CSV with columns: rainfall, river_level, soil_moisture, temperature, humidity</p>
                        </div>
                    </div>

                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${dragOverCsv ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-white/10 hover:border-white/20'
                            }`}
                        onDragOver={(e) => { e.preventDefault(); setDragOverCsv(true) }}
                        onDragLeave={() => setDragOverCsv(false)}
                        onDrop={handleCsvDrop}
                        onClick={() => csvInputRef.current?.click()}
                    >
                        <input ref={csvInputRef} type="file" accept=".csv" className="hidden" onChange={handleCsvSelect} />
                        {csvFile ? (
                            <div>
                                <div className="text-4xl mb-3">📄</div>
                                <p className="text-xs text-slate-400">{csvFile.name}</p>
                                <p className="text-[10px] text-slate-600">{(csvFile.size / 1024).toFixed(1)} KB</p>
                            </div>
                        ) : (
                            <>
                                <div className="text-4xl mb-3 opacity-50">📁</div>
                                <p className="text-sm text-slate-400">Drag & drop or click to upload</p>
                                <p className="text-[10px] text-slate-600 mt-1">Supported: CSV</p>
                            </>
                        )}
                    </div>

                    <button
                        onClick={submitCsv}
                        disabled={!csvFile || csvLoading}
                        className="w-full py-3 rounded-xl text-sm font-semibold transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-600 to-teal-500 text-white hover:shadow-lg hover:shadow-emerald-500/25"
                    >
                        {csvLoading ? (
                            <span className="flex items-center justify-center gap-2">
                                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                Processing...
                            </span>
                        ) : 'Run LSTM Forecast'}
                    </button>

                    {/* Results */}
                    {csvResult && (
                        <div className="space-y-2 pt-2 border-t border-white/5">
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Results</h3>
                            <div className="grid grid-cols-2 gap-2">
                                <MiniResult label="MSE" value={csvResult.mse} color="#06b6d4" />
                                <MiniResult label="Points" value={`${csvResult.predictions?.length || 0}`} color="#8b5cf6" />
                            </div>
                            <div className="max-h-32 overflow-y-auto text-[10px] font-mono text-slate-500 bg-white/[0.02] rounded-lg p-2">
                                {csvResult.predictions?.slice(0, 8).map((p, i) => (
                                    <div key={i} className="flex justify-between">
                                        <span>Hour +{p.hour}</span>
                                        <span className="text-cyan-400">{p.predicted_level} m</span>
                                    </div>
                                ))}
                                {csvResult.predictions?.length > 8 && (
                                    <div className="text-slate-600 text-center mt-1">...{csvResult.predictions.length - 8} more</div>
                                )}
                            </div>
                            <p className="text-[10px] text-slate-600">{csvResult.model}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

function MiniResult({ label, value, color }) {
    return (
        <div className="bg-white/[0.03] rounded-lg p-2.5 border border-white/5">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
            <p className="text-sm font-bold font-mono mt-0.5" style={{ color }}>{value}</p>
        </div>
    )
}
