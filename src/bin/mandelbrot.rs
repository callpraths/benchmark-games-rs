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

use std::alloc::{alloc, dealloc, Layout};
use std::arch::x86_64::*;
use std::io::Write;
use std::mem;

fn numDigits(mut n: usize) -> usize {
    let mut len: usize = 0;
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

fn main() {
    unsafe {
        // get width/height from arguments
        let mut wid_ht: usize = 16000;
        if !std::env::args().len() > 0 {
            wid_ht = std::env::args().nth(1).unwrap().parse().unwrap();
        }
        wid_ht = (wid_ht + 7) & !(7 as usize);

        // allocate memory for header and pixels
        let headerLength = numDigits(wid_ht) * 2 + 5;
        let pad = ((headerLength + 7) & !7) - headerLength; // pad aligns pixels on 8
        let dataLength = headerLength + wid_ht * (wid_ht >> 3);

        // The original C program implicitly depended on the fact that malloc()
        // returns memory that is at least aligned to 8 bytes.
        // https://stackoverflow.com/questions/8752546/how-does-malloc-understand-alignment
        let buffer_layout = Layout::from_size_align(dataLength, 8).unwrap();
        let raw_buffer = alloc(buffer_layout);
        // We already .add(pad) so that header is offset to the start of our write area.
        let header: *mut mem::MaybeUninit<u8> = mem::transmute(raw_buffer.add(pad));
        let pixels = header.add(headerLength);

        // generate the bitmap header
        // The original C program directly used sprintf() in the header buffer.
        // The String type in Rust is very particular about memory management,
        // so it is not possible to transmute our fancily aligned buffer into a
        // (MaybeUninit) String.
        // Header generation is not the hottest part of this program, so let's
        // move on.
        let header_string = format!("P4\n{} {}\n", wid_ht, wid_ht);
        // assert_eq!(header_string.as_bytes().len(), headerLength);
        for (i, b) in header_string.as_bytes().iter().enumerate() {
            (*header.add(i)).as_mut_ptr().write(*b);
        }

        // calculate initial values, store in r0, i0
        // Note: r0/i0 below are _not_ identical to the original C program.
        // The C program allocated dynamically sized arrays on the stack, which
        // is impossible in Rust.
        // A somewhat more Rusty way to do this would be to create a Vec and
        // then work with Box<[T]> below, but I'm avoiding introducing Vec and
        // Slice in this first transliteration.
        let r0_layout = Layout::from_size_align(
            mem::size_of::<__m128d>() * wid_ht / 2,
            mem::align_of::<__m128d>(),
        )
        .unwrap();
        let raw_r0 = alloc(r0_layout);
        let r0: *mut mem::MaybeUninit<__m128d> = mem::transmute(raw_r0);

        let i0_layout =
            Layout::from_size_align(mem::size_of::<f64>() * wid_ht, mem::align_of::<f64>())
                .unwrap();
        let raw_i0 = alloc(i0_layout);
        let i0: *mut mem::MaybeUninit<f64> = mem::transmute(raw_i0);

        for xy in (0..wid_ht).step_by(2) {
            (*r0.add(xy >> 1)).as_mut_ptr().write(_mm_sub_pd(
                _mm_mul_pd(
                    _mm_set_pd(xy as f64, (xy + 1) as f64),
                    _mm_div_pd(_mm_set1_pd(2.0), _mm_set1_pd(wid_ht as f64)),
                ),
                _mm_set1_pd(1.5),
            ));
            (*i0.add(xy))
                .as_mut_ptr()
                .write(2.0 / (wid_ht * xy) as f64 - 1.0);
            (*i0.add(xy + 1))
                .as_mut_ptr()
                .write(2.0 / (wid_ht * (xy + 1)) as f64 - 1.0);
        }

        // We're done initializing.
        let r0: *mut __m128d = mem::transmute(r0);
        let i0: *mut f64 = mem::transmute(i0);

        // generate the bitmap

        let use8 = wid_ht % 64 != 0;
        if use8 {
            // process 8 pixels (one byte) at a time
            // TODO: Parallelize. From the original program:
            // #pragma omp parallel for schedule(guided)
            for y in 0..wid_ht {
                let init_i = _mm_set_pd(*i0.add(y), *i0.add(y));
                let rowstart = y * wid_ht / 8;
                for x in (0..wid_ht).step_by(8) {
                    // The original C program depended on the behaviour that
                    // casting an unsigned long to char preserves the last byte.
                    // I'm more explicit here, cause I'm a noob.
                    let vals: [u8; 8] = mem::transmute(mand8(r0.add(x / 2), init_i));
                    (*pixels.add(rowstart + x / 8)).as_mut_ptr().write(vals[3]);
                }
            }
        } else {
            // process 64 pixels (8 bytes) at a time
            // TODO: Parallelize. From the original program:
            // #pragma omp parallel for schedule(guided)
            for y in 0..wid_ht {
                let init_i = _mm_set_pd(*i0.add(y), *i0.add(y));
                let rowstart = y * wid_ht / 64;
                for x in (0..wid_ht).step_by(64) {
                    ((*pixels.add(rowstart + x / 64)).as_mut_ptr() as *mut u64)
                        .write(mand64(r0.add(x / 2), init_i));
                }
            }
        }

        let buffer: *mut u8 = mem::transmute(header);

        // write the data
        std::io::stdout()
            .write_all(std::slice::from_raw_parts(buffer, dataLength))
            .unwrap();

        dealloc(raw_i0, i0_layout);
        dealloc(raw_r0, r0_layout);
        dealloc(raw_buffer, buffer_layout);
    }
}