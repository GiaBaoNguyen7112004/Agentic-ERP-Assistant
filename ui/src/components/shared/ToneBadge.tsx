import type * as React from 'react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { Tone } from '@/lib/routeBadge'

const toneClasses: Record<Tone, string> = {
  neutral: 'bg-muted text-muted-foreground border-border',
  info: 'bg-info/10 text-info border-info/40',
  success: 'bg-success/10 text-success border-success/40',
  warning: 'bg-warning/10 text-warning border-warning/40',
  destructive: 'bg-destructive/10 text-destructive border-destructive/40',
  memory: 'bg-memory/10 text-memory border-memory/40',
}

/** Badge extended with the project's six tones (§1's semantic maps). */
export function ToneBadge({
  tone,
  className,
  children,
  ...props
}: React.ComponentProps<typeof Badge> & { tone: Tone }) {
  return (
    <Badge variant="outline" className={cn(toneClasses[tone], className)} {...props}>
      {children}
    </Badge>
  )
}