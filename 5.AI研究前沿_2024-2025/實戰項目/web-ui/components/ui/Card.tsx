import { ReactNode } from 'react'
import clsx from 'clsx'

interface CardProps {
  children: ReactNode
  className?: string
  /** 是否有邊框 */
  bordered?: boolean
  /** 是否有懸停效果 */
  hoverable?: boolean
  /** 內距大小 */
  padding?: 'sm' | 'md' | 'lg'
}

const paddingStyles = {
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8'
}

/**
 * 可複用的 Card 容器組件
 */
export function Card({
  children,
  className,
  bordered = false,
  hoverable = true,
  padding = 'md'
}: CardProps) {
  return (
    <div
      className={clsx(
        'bg-white rounded-lg shadow-md',
        paddingStyles[padding],
        bordered && 'border border-gray-200',
        hoverable && 'hover:shadow-lg transition-shadow',
        className
      )}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps {
  children: ReactNode
  className?: string
}

/**
 * Card 標題區域
 */
export function CardHeader({ children, className }: CardHeaderProps) {
  return (
    <div className={clsx('mb-4', className)}>
      {children}
    </div>
  )
}

interface CardTitleProps {
  children: ReactNode
  className?: string
  as?: 'h2' | 'h3' | 'h4'
}

/**
 * Card 標題
 */
export function CardTitle({ children, className, as: Tag = 'h4' }: CardTitleProps) {
  return (
    <Tag className={clsx('text-lg font-semibold text-gray-900', className)}>
      {children}
    </Tag>
  )
}

interface CardContentProps {
  children: ReactNode
  className?: string
}

/**
 * Card 內容區域
 */
export function CardContent({ children, className }: CardContentProps) {
  return (
    <div className={clsx('text-gray-600', className)}>
      {children}
    </div>
  )
}

interface CardFooterProps {
  children: ReactNode
  className?: string
}

/**
 * Card 底部區域
 */
export function CardFooter({ children, className }: CardFooterProps) {
  return (
    <div className={clsx('mt-4 pt-4 border-t border-gray-100', className)}>
      {children}
    </div>
  )
}

export default Card
