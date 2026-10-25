import { useState, useRef, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { predictImage, predictTimeseries, predictSegmentation, getSampleS1Chips, predictSegmentSample } from '../services/api'
import { Link } from 'react-router-dom'

const tabs = [
    { id: 'classifier', label: 'CNN Flood Detector', icon: '🔍' },
    { id: 'segmentation', label: 'Flood Segmentation (U-Net)', icon: '🎨' },
    { id: 'lstm', label: 'LSTM Forecast', icon: '📈' },
]

export default function AIModels() {
    const [activeTab, setActiveTab] = useState('classifier')
    const [classifierResult, setClassifierResult] = useState(null)
    const [segmentationResult, setSegmentationResult] = useState(null)
    const [sampleChips, setSampleChips] = useState([])
    const [selectedChip, setSelectedChip] = useState('')
    const [maskOpacity, setMaskOpacity] = useState(0.85)
    const [lstmData, setLstmData] = useState([])
    const [lstmError, setLstmError] = useState('')
    const [uploading, setUploading] = useState(false)
    const [uploadError, setUploadError] = useState('')
    const [previewUrl, setPreviewUrl] = useState(null)
    const fileInputRef = useRef(null)
    const segInputRef = useRef(null)
    const csvInputRef = useRef(null)

    // Load available Sen1Floods11 sample chips on mount
    useEffect(() => {
        getSampleS1Chips()
            .then(res => {
                if (res.data?.samples) setSampleChips(res.data.samples)
            })
            .catch(err => console.debug('Could not load sample S1 chips:', err))
    }, [])

    const handleTimeseriesUpload = async (event) => {
        const file = event.target.files?.[0]
        if (!file) return
        setUploading(true)
        setLstmError('')
        try {
            const body = new FormData()
            body.append('file', file)
            const { data } = await predictTimeseries(body)
            setLstmData((data.predictions || []).map(point => ({
                hour: `+${point.hour}h`,
                predicted: point.predicted_level,
                upper: point.upper_ci,
                lower: point.lower_ci,
            })))
        } catch (error) {
            setLstmData([])
            setLstmError(error.response?.data?.detail || error.message || 'Forecast failed.')
        } finally {
            setUploading(false)
            event.target.value = ''
        }
    }

    const handleImageUpload = async (e, type) => {
        const file = e.target.files?.[0]
        if (!file) return

        setPreviewUrl(URL.createObjectURL(file))
        setUploading(true)
        setUploadError('')
        try {
            if (type === 'classifier') {
                const body = new FormData()
                body.append('file', file)
                const { data } = await predictImage(body)
                setClassifierResult(data)
            } else {
                const body = new FormData()
                body.append('file', file)
                const { data } = await predictSegmentation(body)
                setSegmentationResult(data)
            }
        } catch (error) {
            setUploadError(error.response?.data?.detail || error.message || 'Image analysis failed.')
            if (type === 'classifier') setClassifierResult(null)
            if (type === 'segmentation') setSegmentationResult(null)
        } finally {
            setUploading(false)
            e.target.value = ''
        }
    }

    const handleSampleChipSelect = async (chipFilename) => {
        if (!chipFilename) return
        setSelectedChip(chipFilename)
        setUploading(true)
        setUploadError('')
        try {
            const { data } = await predictSegmentSample(chipFilename)
            setSegmentationResult(data)
            setPreviewUrl(data.sar_preview_b64)
        } catch (error) {
            setUploadError(error.response?.data?.detail || error.message || 'Sample segmentation failed.')
            setSegmentationResult(null)
        } finally {
            setUploading(false)
        }
    }

    return (
        <div className="p-6 lg:p-8 space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold gradient-text">AI Model Visualization</h1>
                <p className="text-sm text-slate-500 mt-1">Deep learning flood analysis & forecasting models</p>
            </div>
            <Link to="/" className="inline-flex rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 hover:border-cyan-500/40">← Back to Dashboard</Link>

            {/* Tabs */}
            <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === tab.id
                                ? 'bg-gradient-to-r from-blue-600/30 to-cyan-500/20 text-blue-400 border border-blue-500/20 shadow-lg'
                                : 'text-slate-400 hover:text-slate-300 hover:bg-white/5'
                            }`}
                    >
                        <span>{tab.icon}</span>
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="glass-card p-6">
                {activeTab === 'classifier' && (
                    <div className="space-y-6">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-xl">🔍</div>
                            <div>
                                <h2 className="text-lg font-bold text-slate-200">CNN Flood Classifier</h2>
                                <p className="text-xs text-slate-500">Upload a satellite image → get flood probability + bounding box</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Upload area */}
                            <div className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center hover:border-blue-500/30 transition-colors cursor-pointer"
                                onClick={() => fileInputRef.current?.click()}>
                                <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
                                    onChange={(e) => handleImageUpload(e, 'classifier')} />
                                {previewUrl && activeTab === 'classifier' ? (
                                    <div className="relative">
                                        <img src={previewUrl} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                                    </div>
                                ) : (
                                    <>
                                        <div className="text-4xl mb-3">🛰️</div>
                                        <p className="text-sm text-slate-400">Click to upload satellite image</p>
                                        <p className="text-xs text-slate-600 mt-1">PNG, JPG, TIF — 224×224 recommended</p>
                                    </>
                                )}
                            </div>

                            {/* Results */}
                            <div className="space-y-4">
                                {uploading ? (
                                    <div className="flex flex-col items-center justify-center h-full gap-4">
                                        <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                                        <p className="text-sm text-slate-400">Running CNN inference...</p>
                                    </div>
                                ) : classifierResult ? (
                                    <>
                                        <ResultCard label="Flood Probability"
                                            value={`${(classifierResult.flood_probability * 100).toFixed(1)}%`}
                                            color={classifierResult.flood_probability > 0.5 ? '#ef4444' : '#22c55e'} />
                                        <ResultCard label="Decision Margin (uncalibrated)"
                                            value={`${(classifierResult.confidence * 100).toFixed(1)}%`}
                                            color="#3b82f6" />
                                        <ResultCard label="Water Area Estimate"
                                            value={`${(classifierResult.water_area_fraction * 100).toFixed(1)}%`}
                                            color="#8b5cf6" />
                                        <div className="text-[10px] text-slate-600 mt-4">
                                            Model: {classifierResult.model}. Experimental: trained on synthetic imagery, not validated for emergency decisions.
                                        </div>
                                    </>
                                ) : (
                                    <div className="flex items-center justify-center h-full text-slate-600 text-sm">
                                        Upload an image to see results
                                    </div>
                                )}
                                {uploadError && <p className="text-sm text-rose-400" role="alert">{uploadError}</p>}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'segmentation' && (
                    <div className="space-y-6">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-xl">🎨</div>
                                <div>
                                    <h2 className="text-lg font-bold text-slate-200">U-Net Flood Segmentation (Sentinel-1 SAR)</h2>
                                    <p className="text-xs text-slate-500">ResNet-34 Encoder • Calibrated Threshold τ = 0.5517 • Dual-pol VV/VH SAR</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                    ● Model Active
                                </span>
                            </div>
                        </div>

                        {/* Quick Sample Selector */}
                        {sampleChips.length > 0 && (
                            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                                        ⚡ Try Sen1Floods11 Test Chips
                                    </span>
                                    <span className="text-[11px] text-slate-500">Click to run segmentation inference</span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {sampleChips.map((chip) => (
                                        <button
                                            key={chip.filename}
                                            type="button"
                                            onClick={() => handleSampleChipSelect(chip.filename)}
                                            disabled={uploading}
                                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                                                selectedChip === chip.filename
                                                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20 border border-purple-400/40'
                                                    : 'bg-white/5 text-slate-300 hover:bg-white/10 border border-white/10'
                                            }`}
                                        >
                                            {chip.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Upload & Visualizer Column */}
                            <div className="space-y-4">
                                <div
                                    className="border-2 border-dashed border-white/10 rounded-xl p-6 text-center hover:border-purple-500/30 transition-colors cursor-pointer relative overflow-hidden min-h-[280px] flex flex-col items-center justify-center bg-black/20"
                                    onClick={() => segInputRef.current?.click()}
                                >
                                    <input
                                        ref={segInputRef}
                                        type="file"
                                        accept=".tif,.tiff,image/*"
                                        className="hidden"
                                        onChange={(e) => handleImageUpload(e, 'segmentation')}
                                    />

                                    {segmentationResult?.mask_overlay_b64 || previewUrl ? (
                                        <div className="relative w-full h-full flex flex-col items-center justify-center">
                                            {/* SAR Base Image */}
                                            {segmentationResult?.sar_preview_b64 ? (
                                                <img
                                                    src={segmentationResult.sar_preview_b64}
                                                    alt="SAR Preview"
                                                    className="max-h-72 w-auto object-contain rounded-lg shadow-md"
                                                />
                                            ) : previewUrl ? (
                                                <img
                                                    src={previewUrl}
                                                    alt="Preview"
                                                    className="max-h-72 w-auto object-contain rounded-lg shadow-md"
                                                />
                                            ) : null}

                                            {/* Mask Overlay */}
                                            {segmentationResult?.mask_overlay_b64 && (
                                                <img
                                                    src={segmentationResult.mask_overlay_b64}
                                                    alt="Flood Mask Overlay"
                                                    className="absolute inset-0 max-h-72 w-auto mx-auto object-contain rounded-lg pointer-events-none transition-opacity duration-150"
                                                    style={{ opacity: maskOpacity }}
                                                />
                                            )}

                                            <div className="mt-3 flex items-center justify-between w-full px-2">
                                                <span className="text-[11px] text-slate-400">Click to upload another file</span>
                                                <span className="text-[10px] text-purple-300 bg-purple-950/60 border border-purple-500/30 px-2 py-0.5 rounded">
                                                    Cyan = Predicted Flood Waters
                                                </span>
                                            </div>
                                        </div>
                                    ) : (
                                        <>
                                            <div className="text-4xl mb-3">📡</div>
                                            <p className="text-sm font-medium text-slate-300">Click to upload Sentinel-1 SAR or GeoTIFF</p>
                                            <p className="text-xs text-slate-500 mt-1">Supports dual-band GeoTIFF (.tif, .tiff) or SAR imagery</p>
                                        </>
                                    )}
                                </div>

                                {/* Opacity Slider */}
                                {segmentationResult?.mask_overlay_b64 && (
                                    <div className="p-3 rounded-lg bg-white/[0.03] border border-white/5 flex items-center gap-4">
                                        <span className="text-xs text-slate-400 whitespace-nowrap">Mask Opacity:</span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.05"
                                            value={maskOpacity}
                                            onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                                            className="w-full accent-purple-500 cursor-pointer"
                                        />
                                        <span className="text-xs font-mono text-purple-300 w-10 text-right">
                                            {Math.round(maskOpacity * 100)}%
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* Telemetry & Metrics Column */}
                            <div className="space-y-4">
                                {uploading ? (
                                    <div className="flex flex-col items-center justify-center h-64 gap-4 glass-card p-6">
                                        <div className="w-12 h-12 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin"></div>
                                        <p className="text-sm text-slate-300 font-medium">Running U-Net segmentation inference...</p>
                                        <p className="text-xs text-slate-500">Applying dB scaling, VV/VH ratio, and calibrating threshold</p>
                                    </div>
                                ) : segmentationResult ? (
                                    <div className="space-y-3">
                                        <ResultCard
                                            label="Flood Area Coverage"
                                            value={`${((segmentationResult.flood_area_fraction ?? 0) * 100).toFixed(2)}%`}
                                            color="#8b5cf6"
                                        />
                                        <div className="grid grid-cols-2 gap-3">
                                            <ResultCard
                                                label="Flood Pixels"
                                                value={(segmentationResult.flood_pixels ?? 0).toLocaleString()}
                                                color="#06b6d4"
                                            />
                                            <ResultCard
                                                label="Total Pixels"
                                                value={(segmentationResult.total_pixels ?? 0).toLocaleString()}
                                                color="#64748b"
                                            />
                                        </div>
                                        <div className="grid grid-cols-2 gap-3">
                                            <ResultCard
                                                label="Calibrated Threshold"
                                                value={`τ = ${segmentationResult.threshold_used ?? 0.5517}`}
                                                color="#22c55e"
                                            />
                                            <ResultCard
                                                label="Inference Latency"
                                                value={`${Math.round(segmentationResult.inference_latency_ms ?? 0)} ms (${segmentationResult.device ?? 'cpu'})`}
                                                color="#eab308"
                                            />
                                        </div>

                                        {segmentationResult.dimensions && (
                                            <ResultCard
                                                label="Dimensions"
                                                value={`${segmentationResult.dimensions[0]} × ${segmentationResult.dimensions[1]} px`}
                                                color="#3b82f6"
                                            />
                                        )}

                                        {segmentationResult.bounds && (
                                            <div className="glass-card p-3 border border-white/5">
                                                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                                                    WGS84 GeoTIFF Extents
                                                </div>
                                                <div className="text-xs font-mono text-cyan-300">
                                                    [{segmentationResult.bounds.map(b => b.toFixed(4)).join(', ')}]
                                                </div>
                                                <div className="text-[10px] text-slate-500 mt-1">
                                                    [West, South, East, North] • Ready for Map Overlay
                                                </div>
                                            </div>
                                        )}

                                        <div className="text-[11px] text-slate-500 p-3 rounded-lg bg-white/[0.02] border border-white/5 space-y-1">
                                            <div className="text-slate-400 font-medium">Model Architecture:</div>
                                            <div>U-Net (ResNet-34 backbone, pretrained ImageNet weights, BCE+Dice loss). Trained on Sen1Floods11 hand-labeled Sentinel-1 flood chips.</div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="glass-card p-8 flex flex-col items-center justify-center h-64 text-center">
                                        <div className="text-3xl mb-2 text-slate-600">🗺️</div>
                                        <p className="text-slate-400 text-sm font-medium">No segmentation run yet</p>
                                        <p className="text-slate-600 text-xs mt-1">
                                            Select a sample chip above or upload a GeoTIFF / SAR image to visualize pixel-level flood extents.
                                        </p>
                                    </div>
                                )}
                                {uploadError && (
                                    <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400" role="alert">
                                        {uploadError}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'lstm' && (
                    <div className="space-y-6">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-xl">📈</div>
                            <div>
                                <h2 className="text-lg font-bold text-slate-200">LSTM Flood Forecaster</h2>
                        <p className="text-xs text-slate-500">Forecast from the latest 10 rows in a feature-complete gauge CSV</p>
                            </div>
                        </div>

                        <div className="h-80">
                            <button type="button" onClick={() => csvInputRef.current?.click()} className="mb-3 rounded-lg border border-cyan-500/30 px-3 py-2 text-sm text-cyan-300">
                                Upload gauge CSV
                            </button>
                            <input ref={csvInputRef} type="file" accept=".csv,text/csv" className="hidden" onChange={handleTimeseriesUpload} />
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={lstmData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="lstmGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.2} />
                                            <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.1} />
                                            <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                    <XAxis dataKey="hour" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false}
                                        axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} interval={5} />
                                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false}
                                        axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                                    <Tooltip contentStyle={{
                                        background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(6,182,212,0.2)',
                                        borderRadius: '12px', fontSize: '12px', color: '#e2e8f0'
                                    }} />
                                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#ciGradient)" />
                                    <Area type="monotone" dataKey="lower" stroke="none" fill="#0a0e1a" />
                                    <Area type="monotone" dataKey="predicted" stroke="#06b6d4" fill="url(#lstmGradient)"
                                        strokeWidth={2.5} dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        {lstmError && <p className="text-sm text-rose-400" role="alert">{lstmError}</p>}
                        {lstmData.length === 0 && !lstmError && <p className="text-sm text-slate-500">Upload a CSV with the model feature columns to run a forecast.</p>}
                    </div>
                )}
            </div>
        </div>
    )
}

function ResultCard({ label, value, color }) {
    return (
        <div className="glass-card p-4 flex items-center justify-between" style={{ borderColor: `${color}20` }}>
            <span className="text-xs text-slate-500 uppercase tracking-wider font-medium">{label}</span>
            <span className="text-lg font-bold font-mono" style={{ color }}>{value}</span>
        </div>
    )
}
