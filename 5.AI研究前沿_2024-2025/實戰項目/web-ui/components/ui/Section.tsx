import { ReactNode } from 'react'
import clsx from 'clsx'

interface SectionProps {
  id?: string
  children: ReactNode
  className?: string
  /** 是否使用白色背景 */
  white?: boolean
  /** 是否添加垂直內距 */
  withPadding?: boolean
}

/**
 * 可複用的 Section 容器組件
 *
 * 提供一致的最大寬度、水平內距和可選的背景色
 */
export function Section({
  id,
  children,
  className,
  white = false,
  withPadding = true
}: SectionProps) {
  return (
    <section
      id={id}
      className={clsx(
        'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8',
        withPadding && 'py-16',
        white && 'bg-white',
        className
      )}
    >
      {children}
    </section>
  )
}

interface SectionHeaderProps {
  title: string
  subtitle?: string
  className?: string
}

/**
 * Section 標題組件
 *
 * 提供一致的標題和副標題樣式
 */
export function SectionHeader({ title, subtitle, className }: SectionHeaderProps) {
  return (
    <div className={clsx('text-center mb-12', className)}>
      <h3 className="text-3xl font-bold text-gray-900">{title}</h3>
      {subtitle && (
        <p className="mt-4 text-lg text-gray-600">{subtitle}</p>
      )}
    </div>
  )
}

export default Section
