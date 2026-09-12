import { ShieldCheck } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { ScopeChip } from '@/components/shared/ScopeChip'
import { SectionHeading } from '@/components/shared/SectionHeading'
import type { UserSummary } from '../protocol'

export function ActorSwitcher({
  users,
  actor,
  onSelect,
}: {
  users: UserSummary[]
  actor: string | null
  onSelect: (actor: string) => void
}) {
  const current = users.find((user) => user.actor === actor)

  return (
    <section className="space-y-2">
      <SectionHeading>Acting as</SectionHeading>
      <Select value={actor ?? ''} onValueChange={onSelect}>
        <SelectTrigger className="w-full" aria-label="Actor">
          <SelectValue placeholder="Choose an actor" />
        </SelectTrigger>
        <SelectContent>
          {users.map((user) => (
            <SelectItem key={user.actor} value={user.actor}>
              {user.display_name} — {user.role}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {current && (
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline" className="font-mono text-xs">
            {current.project_code}
          </Badge>
          {current.can_approve && (
            <ToneBadge tone="success">
              <ShieldCheck aria-hidden />
              can approve
            </ToneBadge>
          )}
          {current.scopes.map((scope) => (
            <ScopeChip key={scope} scope={scope} />
          ))}
        </div>
      )}
    </section>
  )
}