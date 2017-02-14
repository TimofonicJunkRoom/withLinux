module Main where

-- https://en.wikipedia.org/wiki/Haskell_%28programming_language%29
-- http://learnyouahaskell.com

factorial n
  | n < 2     = 1
  | otherwise = n * factorial (n-1)

main :: IO ()
main = do
  putStrLn "hello world!"
