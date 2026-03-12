// ResultPanel.tsx
// Shows detection results, heatmap, and severity scores

import SeverityBadge from './SeverityBadge';
import { AnalysisResult, getImageUrl } from '../api/client';

interface Props {
  result: AnalysisResult;
}

export default function ResultPanel({ result }: Props) {
  return (
    <div className="mt-8 space-y-6">

      {/* Overall Score Banner */}
      <div className="bg-gray-900 rounded-2xl p-6 flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-1">Overall Assessment</p>
          <h2 className="text-2xl font-bold text-white">
            {result.total_damages} damage{result.total_damages !== 1 ? 's' : ''} detected
          </h2>
          <p className="text-gray-500 text-sm mt-1">
            Processed in {result.processing_time_ms}ms
          </p>
        </div>
        <SeverityBadge
          label={result.overall_severity}
          score={result.overall_score}
          size="lg"
        />
      </div>

      {/* Images Side by Side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* Detection Image */}
        <div className="bg-gray-900 rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
            <span>🔍</span> Detection Results
          </h3>
          <img
            src={getImageUrl(result.annotated_url)}
            alt="Annotated"
            className="w-full rounded-xl object-contain max-h-80"
          />
        </div>

        {/* Heatmap */}
        <div className="bg-gray-900 rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
            <span>🔥</span> Explainability Heatmap
            <span className="text-xs text-gray-600 ml-1">(red = AI focus areas)</span>
          </h3>
          <img
            src={getImageUrl(result.heatmap_url)}
            alt="Heatmap"
            className="w-full rounded-xl object-contain max-h-80"
          />
        </div>
      </div>

      {/* Individual Detections */}
      {result.detections.length > 0 && (
        <div className="bg-gray-900 rounded-2xl p-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-4 flex items-center gap-2">
            <span>📋</span> Damage Breakdown
          </h3>
          <div className="space-y-3">
            {result.detections.map((det, i) => (
              <div
                key={i}
                className="bg-gray-800 rounded-xl p-4 flex items-center justify-between"
              >
                <div className="flex-1">
                  <p className="font-semibold text-white capitalize">
                    {det.damage_type.replace(/_/g, ' ')}
                  </p>
                  <div className="flex items-center gap-4 mt-1">
                    <p className="text-xs text-gray-500">
                      Confidence: {(det.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">
                      Area: {(det.explanation.area_ratio * 100).toFixed(1)}% of image
                    </p>
                    <p className="text-xs text-gray-500">
                      Edges: {(det.explanation.edge_density * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <SeverityBadge
                  label={det.severity_label}
                  score={det.severity_score}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No damage case */}
      {result.detections.length === 0 && (
        <div className="bg-gray-900 rounded-2xl p-8 text-center">
          <p className="text-4xl mb-3">✅</p>
          <p className="text-white font-semibold text-lg">No damage detected</p>
          <p className="text-gray-400 text-sm mt-1">
            The vehicle appears to be in good condition
          </p>
        </div>
      )}
    </div>
  )
}