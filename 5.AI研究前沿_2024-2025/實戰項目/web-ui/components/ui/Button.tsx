import { ReactNode, ButtonHTMLAttributes } from 'react'
import Link from 'next/link'
import clsx from 'clsx'

type ButtonVariant = 'primary' | 'secondary' | 'outline'
type ButtonSize = 'sm' | 'md' | 'lg'

interface BaseButtonProps {
  variant?: ButtonVariant
  size?: ButtonSize
  children: ReactNode
  className?: string
}

interface ButtonAsButtonProps extends BaseButtonProps, ButtonHTMLAttributes<HTMLButtonElement> {
  href?: never
}

interface ButtonAsLinkProps extends BaseButtonProps {
  href: string
  onClick?: never
}

type ButtonProps = ButtonAsButtonProps | ButtonAsLinkProps

const variantStyles: Record<ButtonVariant, string> = {
  primary: 'text-white bg-blue-600 hover:bg-blue-700 border-transparent',
  secondary: 'text-gray-700 bg-white hover:bg-gray-50 border-gray-300',
  outline: 'text-blue-600 bg-transparent hover:bg-blue-50 border-blue-600'
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-base'
}

/**
 * 可複用的 Button 組件
 *
 * 支持多種變體和尺寸，可以作為按鈕或連結使用
 *
 * @example
 * ```tsx
 * <Button variant="primary">點擊我</Button>
 * <Button href="/about" variant="secondary">了解更多</Button>
 * ```
 */
export function Button({
  variant = 'primary',
  size = 'md',
  children,
  className,
  ...props
}: ButtonProps) {
  const baseStyles = clsx(
    'inline-flex items-center justify-center font-medium rounded-md border transition-colors',
    variantStyles[variant],
    sizeStyles[size],
    className
  )

  if ('href' in props && props.href) {
    return (
      <Link href={props.href} className={baseStyles}>
        {children}
      </Link>
    )
  }

  return (
    <button className={baseStyles} {...(props as ButtonAsButtonProps)}>
      {children}
    </button>
  )
}

export default Button
