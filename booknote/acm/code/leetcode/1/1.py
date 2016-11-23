class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # note, index starts from 0

        #for (i, vi) in enumerate(nums):
        #    for (j, vj) in enumerate(nums):
        #        # don't add to itself
        #        if i == j: continue
        #        if vi + vj == target: return [i, j]
        #return [-1, -1]
        # => Time Limit Exceeded

        #nlen = len(nums)
        #for (i, vi) in enumerate(nums):
        #    for (j, vj) in enumerate(reversed(nums)):
        #        if i == nlen-j-1:
        #            continue
        #        elif vi+vj==target:
        #            return [i, nlen-j-1]
        #return [-1, -1]
        # => Time Limit Exceeded
