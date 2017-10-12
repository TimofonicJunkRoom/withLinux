class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        loc = dict((v, i) for i, v in enumerate(nums)) # O(n)

        for i, v in enumerate(nums): # O(n)
            if loc.get(target-v, False):
                j = loc.get(target-v)
                return [i, j]
        '''
        vtoi = dict()
        for i, v in enumerate(nums):
            #print(i, v, vtoi)
            idx = vtoi.get(target - v, None)
            #print('idx', target-v, idx)
            if None!=idx: return [i, idx]
            else: vtoi[v] = i
        return False;  # O(n), expect O(n/2)

s = Solution()
print(s.twoSum([3,3], 6))
print(s.twoSum([2,7,11,15], 13))

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
