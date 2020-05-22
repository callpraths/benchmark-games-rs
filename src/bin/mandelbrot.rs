// The Computer Language Benchmarks Game
// https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
//
// Contributed by Prathmesh Prabhu
// based on the C program contributed by Kevin Miller
// inspired by the C-to-Rust conversion described at
// http://cliffle.com/p/dangerust/

// Temporary
#![allow(dead_code, unused_variables)]
// Will outlast current iteration.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use std::arch::x86_64::*;
use std::mem;

fn numDigits(mut n: u64) -> u64 {
    let mut len: u64 = 0;
    while n != 0 {
        n = n / 10;
        len += 1;
    }
    len
}

#[inline(always)]
unsafe fn vec_nle(v: *mut __m128d, f: f64) -> i64 {
    // TODO: https://github.com/searchivarius/BlogCode/blob/master/2014/5/14/mm_extract_pd.cpp
    // is "more correct" and sometimes faster.
    // TODO: Also see gcc option -mfpmath=sse which would lead to normal
    // floating point operations also using SSE registers. This could lead to
    // more efficient type punning.
    // https://stackoverflow.com/questions/12624466/get-member-of-m128-by-index
    if *(v.add(0) as *mut f64).add(0) <= f
        || *(v.add(0) as *mut f64).add(1) <= f
        || *(v.add(1) as *mut f64).add(0) <= f
        || *(v.add(1) as *mut f64).add(1) <= f
        || *(v.add(2) as *mut f64).add(0) <= f
        || *(v.add(2) as *mut f64).add(1) <= f
        || *(v.add(3) as *mut f64).add(0) <= f
        || *(v.add(3) as *mut f64).add(1) <= f
    {
        0
    } else {
        -1
    }
}

#[inline(always)]
unsafe fn clrPixels_nle(v: *mut __m128d, f: f64, pix8: *mut u64) {
    if !(*(v.add(0) as *mut f64).add(0) <= f) {
        *pix8 &= 0x7f
    }
    if !(*(v.add(0) as *mut f64).add(1) <= f) {
        *pix8 &= 0xbf
    }
    if !(*(v.add(1) as *mut f64).add(0) <= f) {
        *pix8 &= 0xdf
    }
    if !(*(v.add(1) as *mut f64).add(1) <= f) {
        *pix8 &= 0xef
    }
    if !(*(v.add(2) as *mut f64).add(0) <= f) {
        *pix8 &= 0xf7
    }
    if !(*(v.add(2) as *mut f64).add(1) <= f) {
        *pix8 &= 0xfb
    }
    if !(*(v.add(3) as *mut f64).add(0) <= f) {
        *pix8 &= 0xfd
    }
    if !(*(v.add(3) as *mut f64).add(1) <= f) {
        *pix8 &= 0xfe
    }
}

#[inline(always)]
unsafe fn calcSum(
    r: *mut __m128d,
    i: *mut __m128d,
    sum: *mut mem::MaybeUninit<__m128d>,
    init_r: *mut __m128d,
    init_i: __m128d,
) {
    for pair in 0..4 {
        let r2: __m128d = _mm_mul_pd(*r.add(pair), *r.add(pair));
        let i2: __m128d = _mm_mul_pd(*i.add(pair), *i.add(pair));
        let ri: __m128d = _mm_mul_pd(*r.add(pair), *i.add(pair));
        (*sum.add(pair)).as_mut_ptr().write(_mm_add_pd(r2, i2));
        *r.add(pair) = _mm_add_pd(_mm_sub_pd(r2, i2), *init_r.add(pair));
        *i.add(pair) = _mm_add_pd(_mm_add_pd(ri, ri), init_i);
    }
}

#[inline(always)]
unsafe fn mand8(init_r: *mut __m128d, init_i: __m128d) -> u64 {
    let mut r = [mem::MaybeUninit::<__m128d>::uninit(); 4];
    let mut i = [mem::MaybeUninit::<__m128d>::uninit(); 4];
    let mut sum = [mem::MaybeUninit::<__m128d>::uninit(); 4];
    for pair in 0..4 {
        r[pair].as_mut_ptr().write(*init_r.add(pair));
        i[pair].as_mut_ptr().write(init_i);
    }
    let mut r: [__m128d; 4] = mem::transmute(r);
    let mut i: [__m128d; 4] = mem::transmute(i);

    let mut pix8: u64 = 0xff;
    for j in 0..6 {
        for k in 0..8 {
            calcSum(
                r.as_mut_ptr(),
                i.as_mut_ptr(),
                sum.as_mut_ptr(),
                init_r,
                init_i,
            );
        }
        if vec_nle(sum.as_mut_ptr() as *mut __m128d, 4.0) != 0 {
            pix8 = 0x00;
            break;
        }
    }
    if pix8 != 0 {
        calcSum(
            r.as_mut_ptr(),
            i.as_mut_ptr(),
            sum.as_mut_ptr(),
            init_r,
            init_i,
        );
        calcSum(
            r.as_mut_ptr(),
            i.as_mut_ptr(),
            sum.as_mut_ptr(),
            init_r,
            init_i,
        );
        clrPixels_nle(sum.as_mut_ptr() as *mut __m128d, 4.0, &mut pix8);
    }
    return pix8;
}

unsafe fn mand64(mut init_r: *mut __m128d, init_i: __m128d) -> u64 {
    let mut pix64: u64 = 0;

    for byte in 0..8 {
        let pix8 = mand8(init_r, init_i);

        pix64 = (pix64 >> 8) | (pix8 << 56);
        init_r = init_r.add(4);
    }

    return pix64;
}

fn main() {}

// This C-to-Rust conversion is WIP.
// The remaining C program, commented out, follows.
/*

int main(int argc, char ** argv)
{
    // get width/height from arguments

    long wid_ht = 16000;
    if (argc >= 2)
    {
        wid_ht = atoi(argv[1]);
    }
    wid_ht = (wid_ht+7) & ~7;


    // allocate memory for header and pixels

    long headerLength = numDigits(wid_ht)*2+5;
    long pad = ((headerLength + 7) & ~7) - headerLength; // pad aligns pixels on 8
    long dataLength = headerLength + wid_ht*(wid_ht>>3);
    unsigned char * const buffer = malloc(pad + dataLength);
    unsigned char * const header = buffer + pad;
    unsigned char * const pixels = header + headerLength;


    // generate the bitmap header

    sprintf((char *)header, "P4\n%ld %ld\n", wid_ht, wid_ht);


    // calculate initial values, store in r0, i0

    __m128d r0[wid_ht/2];
    double i0[wid_ht];

    for(long xy=0; xy<wid_ht; xy+=2)
    {
        r0[xy>>1] = 2.0 / wid_ht * (__m128d){xy,  xy+1} - 1.5;
        i0[xy]    = 2.0 / wid_ht *  xy    - 1.0;
        i0[xy+1]  = 2.0 / wid_ht * (xy+1) - 1.0;
    }


    // generate the bitmap

    long use8 = wid_ht%64;
    if (use8)
    {
        // process 8 pixels (one byte) at a time
        #pragma omp parallel for schedule(guided)
        for(long y=0; y<wid_ht; y++)
        {
            __m128d init_i = (__m128d){i0[y], i0[y]};
            long rowstart = y*wid_ht/8;
            for(long x=0; x<wid_ht; x+=8)
            {
                pixels[rowstart + x/8] = mand8(r0+x/2, init_i);
            }
        }
    }
    else
    {
        // process 64 pixels (8 bytes) at a time
        #pragma omp parallel for schedule(guided)
        for(long y=0; y<wid_ht; y++)
        {
            __m128d init_i = (__m128d){i0[y], i0[y]};
            long rowstart = y*wid_ht/64;
            for(long x=0; x<wid_ht; x+=64)
            {
                ((unsigned long *)pixels)[rowstart + x/64] = mand64(r0+x/2, init_i);
            }
        }
    }

    // write the data

    long ret = ret = write(STDOUT_FILENO, header, dataLength);


    free(buffer);

    return 0;
}
*/
