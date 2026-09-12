import { useState, type ReactNode } from 'react'
import { Activity, Bot, PanelLeft } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Spinner } from '@/components/shared/Spinner'

/**
 * The three-column frame: sidebar · chat · trace. Each column scrolls its own
 * overflow (`min-h-0 overflow-y-auto`) so the composer stays pinned -- the
 * scrolling bug ADR 0015 documents; do not replace `min-h-0` with ScrollArea.
 * Below `lg` the chat fills the screen and the sidebar and trace panel open
 * as Sheets from the top bar.
 */
export function AppShell({
  sidebar,
  main,
  aside,
  loading,
}: {
  sidebar?: ReactNode
  main?: ReactNode
  aside?: ReactNode
  loading?: boolean
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [traceOpen, setTraceOpen] = useState(false)

  if (loading) {
    return (
      <div className="grid h-dvh place-items-center bg-background">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Spinner />
          Loading actors…
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-dvh min-h-0 flex-col bg-background">
      {/* top bar, narrow viewports only */}
      <header className="flex items-center gap-1 border-b bg-sidebar px-2 py-1.5 lg:hidden">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Open sidebar"
          onClick={() => setSidebarOpen(true)}
        >
          <PanelLeft aria-hidden />
        </Button>
        <div className="flex items-center gap-1.5 px-1 text-sm font-semibold">
          <Bot className="size-4" aria-hidden />
          Agentic ERP Assistant
        </div>
        <div className="ml-auto">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Open trace"
            onClick={() => setTraceOpen(true)}
          >
            <Activity aria-hidden />
          </Button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:h-full lg:grid-cols-[280px_minmax(0,1fr)_360px]">
        <nav className="hidden min-h-0 flex-col overflow-y-auto border-r bg-sidebar p-4 lg:flex">
          {sidebar}
        </nav>
        {main}
        <aside className="hidden min-h-0 flex-col overflow-y-auto border-l bg-sidebar p-4 lg:flex">
          {aside}
        </aside>
      </div>

      <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetContent side="left" className="w-80 gap-0 overflow-y-auto p-4 sm:max-w-80">
          <SheetHeader>
            <SheetTitle className="sr-only">Sidebar</SheetTitle>
            <div className="flex items-center gap-2 pr-8 text-sm font-semibold">
              <Bot className="size-4" aria-hidden />
              Agentic ERP Assistant
              <Badge variant="outline" className="text-[10px]">
                dev
              </Badge>
            </div>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">{sidebar}</div>
        </SheetContent>
      </Sheet>

      <Sheet open={traceOpen} onOpenChange={setTraceOpen}>
        <SheetContent side="right" className="w-[min(90vw,360px)] gap-0 overflow-y-auto p-4">
          <SheetHeader>
            <SheetTitle className="sr-only">Trace</SheetTitle>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">{aside}</div>
        </SheetContent>
      </Sheet>
    </div>
  )
}