def main():
    from pytools import \
            generate_decreasing_nonnegative_tuples_summing_to, \
            generate_unique_permutations
    for t in generate_decreasing_nonnegative_tuples_summing_to(2, 3):
        print "BASE", t
        #for tperm in generate_unique_permutations(t):
            #print tperm




if __name__ == "__main__":
    main()
