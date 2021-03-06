// The Computer Language Benchmarks Game
// https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
//
// Contributed by Prathmesh Prabhu
// based on the C program contributed by Kevin Miller
// inspired by the C-to-Rust conversion described at
// http://cliffle.com/p/dangerust/

// Will outlast current iteration.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use itertools::iproduct;
use rayon::prelude::*;
use std::arch::x86_64::__m128d;
use std::io::Write;

#[cfg(target_feature = "sse2")]
mod mm {
    use std::arch::x86_64::*;

    pub fn extract_lower(v: __m128d) -> f64 {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0b0000_00000)) }
    }

    pub fn extract_upper(v: __m128d) -> f64 {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0b0000_00001)) }
    }

    pub fn mul_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_mul_pd(a, b) }
    }

    pub fn div_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_div_pd(a, b) }
    }

    pub fn add_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_add_pd(a, b) }
    }

    pub fn sub_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_sub_pd(a, b) }
    }

    pub fn set_pd(a: f64, b: f64) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_set_pd(a, b) }
    }

    pub fn set1_pd(a: f64) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_set1_pd(a) }
    }
}

fn vec_nle(v: &[__m128d; 4], f: f64) -> bool {
    // https://github.com/searchivarius/BlogCode/blob/master/2014/5/14/mm_extract_pd.cpp
    // is "more correct" and sometimes faster than the original C programs
    // indexing of f64 values within an __m128d.

    // TODO: Also see gcc option -mfpmath=sse which would lead to normal
    // floating point operations also using SSE registers. This could lead to
    // more efficient type punning.
    // https://stackoverflow.com/questions/12624466/get-member-of-m128-by-index

    // Rust compiler had trouble unrolling loop here, which led to a noticeable
    // performance hit, so unroll manually.
    mm::extract_lower(v[0]) > f
        && mm::extract_upper(v[0]) > f
        && mm::extract_lower(v[1]) > f
        && mm::extract_upper(v[1]) > f
        && mm::extract_lower(v[2]) > f
        && mm::extract_upper(v[2]) > f
        && mm::extract_lower(v[3]) > f
        && mm::extract_upper(v[3]) > f
}

fn clrPixels_nle(v: &[__m128d; 4], f: f64, pix8: &mut u8) {
    if !(mm::extract_lower(v[0]) <= f) {
        *pix8 &= 0b0111_1111; // 0x7f
    }
    if !(mm::extract_upper(v[0]) <= f) {
        *pix8 &= 0b1011_1111; // 0xbf
    }
    if !(mm::extract_lower(v[1]) <= f) {
        *pix8 &= 0b1101_1111; // 0xdf
    }
    if !(mm::extract_upper(v[1]) <= f) {
        *pix8 &= 0b1110_1111; // 0xef
    }
    if !(mm::extract_lower(v[2]) <= f) {
        *pix8 &= 0b1111_0111; // 0xf7
    }
    if !(mm::extract_upper(v[2]) <= f) {
        *pix8 &= 0b1111_1011; // 0xfb
    }
    if !(mm::extract_lower(v[3]) <= f) {
        *pix8 &= 0b1111_1101; // 0xfd
    }
    if !(mm::extract_upper(v[3]) <= f) {
        *pix8 &= 0b1111_1110; // 0xfe
    }
}

fn calcSum(
    r: &mut [__m128d; 4],
    i: &mut [__m128d; 4],
    sum: &mut [__m128d; 4],
    init_r: &[__m128d; 4],
    init_i: __m128d,
) {
    for idx in 0..4 {
        // Create local variables to avoid overly broad unsafe blocks.
        let cur_r = r[idx];
        let cur_i = i[idx];
        let cur_init_r = init_r[idx];

        let r2: __m128d = mm::mul_pd(cur_r, cur_r);
        let i2: __m128d = mm::mul_pd(cur_i, cur_i);
        let ri: __m128d = mm::mul_pd(cur_r, cur_i);
        sum[idx] = mm::add_pd(r2, i2);
        r[idx] = mm::add_pd(mm::sub_pd(r2, i2), cur_init_r);
        i[idx] = mm::add_pd(mm::add_pd(ri, ri), init_i);
    }
}

fn mand8(init_r: &[__m128d; 4], init_i: __m128d) -> u8 {
    let mut r = *init_r;
    let mut i = [init_i, init_i, init_i, init_i];
    let zero = mm::set1_pd(0.0);
    let mut sum = [zero; 4];

    let mut pix8: u8 = 0xff;
    for _ in 0..6 {
        for _ in 0..8 {
            calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        }

        if vec_nle(&sum, 4.0) {
            pix8 = 0x00;
            break;
        }
    }
    if pix8 != 0 {
        calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        clrPixels_nle(&sum, 4.0, &mut pix8);
    }
    return pix8;
}

fn calc_init_r_pair(x: f64, wid_ht: f64) -> __m128d {
    mm::sub_pd(
        mm::mul_pd(
            // NB: mm::set_pd() reverses the order of arguments when packing
            // f64 into a __m128d.
            // https://stackoverflow.com/questions/5237961/why-does-does-sse-set-mm-set-ps-reverse-the-order-of-arguments
            mm::set_pd(x + 1.0, x),
            mm::div_pd(mm::set1_pd(2.0), mm::set1_pd(wid_ht)),
        ),
        mm::set1_pd(1.5),
    )
}

fn calc_init_r_chunk(x: f64, wid_ht: f64) -> [__m128d; 4] {
    [
        calc_init_r_pair(x as f64, wid_ht as f64),
        calc_init_r_pair((x + 2.0) as f64, wid_ht as f64),
        calc_init_r_pair((x + 4.0) as f64, wid_ht as f64),
        calc_init_r_pair((x + 6.0) as f64, wid_ht as f64),
    ]
}

fn main() {
    // get width/height from arguments
    let mut wid_ht: usize = 16000;
    if !std::env::args().len() > 0 {
        wid_ht = std::env::args().nth(1).unwrap().parse().unwrap();
    }
    wid_ht = (wid_ht + 7) & !(7 as usize);

    std::io::stdout()
        .write_all(format!("P4\n{} {}\n", wid_ht, wid_ht).as_bytes())
        .unwrap();

    let i0 = {
        let mut i0: Vec<__m128d> = Vec::with_capacity(wid_ht);
        for y in 0..wid_ht {
            let v = (2.0 / wid_ht as f64) * y as f64 - 1.0;
            i0.push(mm::set_pd(v, v));
        }
        i0
    };

    // process 8 pixels (one byte) at a time
    let r0 = {
        let mut r0: Vec<[__m128d; 4]> = Vec::with_capacity(wid_ht / 8);
        for x in (0..wid_ht).step_by(8) {
            r0.push(calc_init_r_chunk(x as f64, wid_ht as f64));
        }
        r0
    };

    if std::env::var("PARALLELIZE").is_ok() {
        let pixels: Vec<Vec<u8>> = i0
            .par_iter()
            .map(|i| (&r0).par_iter().map(|r| mand8(r, *i)).collect())
            .collect();
        for row in pixels {
            std::io::stdout().write_all(row.as_slice()).unwrap();
        }
    } else {
        let mut pixels: Vec<u8> = Vec::with_capacity(wid_ht * (wid_ht >> 3));
        pixels.extend(iproduct!(i0.iter(), r0.iter()).map(|(i, r)| mand8(&*r, *i)));
        std::io::stdout().write_all(pixels.as_slice()).unwrap();
    }
}
