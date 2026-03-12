// SeverityBadge.tsx
// Shows severity label with color coding

interface Props {
  label: string;
  score: number;
  size?: 'sm' | 'lg';
}

export default function SeverityBadge({ label, score, size = 'sm' }: Props) {
  const styles: Record<string, string> = {
    'Severe':    'bg-red-500/20 text-red-400 border border-red-500/40',
    'Moderate':  'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40',
    'Minor':     'bg-green-500/20 text-green-400 border border-green-500/40',
    'No Damage': 'bg-blue-500/20 text-blue-400 border border-blue-500/40',
  }

  const icons: Record<string, string> = {
    'Severe':    '🔴',
    'Moderate':  '🟡',
    'Minor':     '🟢',
    'No Damage': '✅',
  }

  const sizeClass = size === 'lg'
    ? 'px-4 py-2 text-base font-bold rounded-xl'
    : 'px-3 py-1 text-sm font-semibold rounded-lg'

  return (
    <span className={`${styles[label] || styles['Minor']} ${sizeClass} inline-flex items-center gap-1.5`}>
      <span>{icons[label] || '🟢'}</span>
      <span>{label}</span>
      <span className="opacity-70">· {score}/100</span>
    </span>
  )
}