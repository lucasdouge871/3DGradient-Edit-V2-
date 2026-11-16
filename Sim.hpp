#pragma once

// Hitbox-based simulator for click-to-jump gameplay (cube and robot).
//
// The Auto Path bot never guesses from fixed distances. It takes the real hitbox
// rectangles of the level objects and simulates the player's hitbox:
//   roll(): what happens if the bot does NOTHING (this is how it knows a jump is needed,
//           and how it knows a jump pad will already handle an obstacle)
//   jump(): what happens if it jumps from a given spot
// then picks the middle of the widest safe stretch of start positions.
//
// This file has no Geode dependency so it can be tested on its own.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace acm {

    struct Rect {
        float minX = 0.f, minY = 0.f, maxX = 0.f, maxY = 0.f;

        bool overlaps(Rect const& o) const {
            return minX < o.maxX && maxX > o.minX && minY < o.maxY && maxY > o.minY;
        }
    };

    constexpr uint32_t kNoId = 0xFFFFFFFFu;

    // A hitbox in the level the bot has to deal with.
    //   hazard : spike, saw, animated hazard...   (kills on touch of the outer hitbox)
    //   solid  : block                            (kills only on the small inner hitbox)
    //   pad    : jump pad                         (launches the player; never kills)
    struct Obstacle {
        float minX = 0.f, maxX = 0.f, minY = 0.f, maxY = 0.f;
        bool hazard = false;
        bool pad = false;
        float padVel = 0.f;      // launch velocity (units per 60 Hz frame, "up" = positive)
        bool padStop = false;    // gravity / spider pad: the sim stops here, the pad takes over
        uint32_t id = kNoId;     // index in the bot's obstacle list

        float refX() const { return hazard ? (minX + maxX) * 0.5f : minX; }
        Rect rect() const { return Rect{minX, minY, maxX, maxY}; }
    };

    namespace sim {

        // Half sizes of the player's hitboxes. The outer box kills on hazards and lands on
        // blocks. Only the small inner box kills on a block, which is why you can graze
        // the corner of a block and live.
        struct Box {
            float hw = 15.f, hh = 15.f;
            float ihw = 4.5f, ihh = 4.5f;
        };

        struct Params {
            float speed = 5.1949f;        // level units per 60 Hz frame (1x speed)
            float jumpVel = 11.180032f;   // cube jump velocity, units per frame
            float gravity = 0.958199f;    // units per frame squared
            float floorY = 0.f;           // y of the player's feet while standing
        };

        struct Result {
            bool safe = false;
            bool padHit = false;          // a jump pad was touched along the way
            float failX = 0.f;
            uint32_t hitId = kNoId;       // which hitbox ended the run
            std::vector<Rect> trail;      // hitbox positions along the run (for drawing)
        };

        struct Window {
            bool found = false;
            float lo = 0.f, hi = 0.f, center = 0.f;
        };

        // Runs the player forward from x = cx. With `startWithJump` it presses jump at the
        // start, otherwise it just rolls along. The run ends safely once the player has
        // landed and is past `clearX`, or unsafely on the first deadly hitbox.
        inline Result run(float cx, Params const& pr, Box const& b, std::vector<Obstacle> const& obs,
                          float clearX, bool startWithJump, bool wantTrail) {
            Result res;
            constexpr float dt = 0.5f;            // half a 60 Hz frame per sub-step
            float const kSlop = b.hh - b.ihh;     // feet this far under a block top still land on it

            float fy = pr.floorY;
            float vy = startWithJump ? pr.jumpVel : 0.f;
            bool airborne = startWithJump;
            std::vector<char> padUsed(obs.size(), 0);

            for (int step = 0; step < 480; ++step) {
                float prevFoot = fy;
                if (airborne) {
                    fy += vy * dt - 0.5f * pr.gravity * dt * dt;
                    vy -= pr.gravity * dt;
                }
                cx += pr.speed * dt;

                // What is under the player: the base floor, or the top of a block we are over.
                float floorY = pr.floorY;
                for (auto const& o : obs) {
                    if (o.hazard || o.pad) continue;
                    if (cx + b.hw > o.minX && cx - b.hw < o.maxX && prevFoot >= o.maxY - kSlop) {
                        floorY = std::max(floorY, o.maxY);
                    }
                }
                if (!airborne && fy > floorY + 0.5f) {
                    airborne = true;              // rolled off the edge of a block
                    vy = 0.f;
                } else if (airborne && fy <= floorY && vy <= 0.f) {
                    fy = floorY;                  // landed
                    vy = 0.f;
                    airborne = false;
                }

                Rect outer{cx - b.hw, fy, cx + b.hw, fy + 2.f * b.hh};
                Rect inner{cx - b.ihw, fy + b.hh - b.ihh, cx + b.ihw, fy + b.hh + b.ihh};

                if (wantTrail && step % 6 == 0) res.trail.push_back(outer);

                // Jump pads launch the player when the hitbox touches them.
                for (size_t i = 0; i < obs.size(); ++i) {
                    auto const& o = obs[i];
                    if (!o.pad || padUsed[i] || !outer.overlaps(o.rect())) continue;
                    padUsed[i] = 1;
                    res.padHit = true;
                    if (o.padStop) {              // gravity / spider pads: not modelled, stop here
                        res.safe = true;
                        return res;
                    }
                    vy = o.padVel;
                    airborne = true;
                }

                for (auto const& o : obs) {
                    if (o.pad) continue;
                    bool dead = o.hazard ? outer.overlaps(o.rect()) : inner.overlaps(o.rect());
                    if (dead) {
                        res.safe = false;
                        res.failX = cx;
                        res.hitId = o.id;
                        return res;
                    }
                }

                if (!airborne && cx - b.hw > clearX) {
                    res.safe = true;
                    return res;
                }
            }
            return res;   // never got past the obstacle
        }

        // Press jump with the player's centre at x = cx.
        inline Result jump(float cx, Params const& pr, Box const& b, std::vector<Obstacle> const& obs,
                           float clearX, bool wantTrail) {
            return run(cx, pr, b, obs, clearX, true, wantTrail);
        }

        // Do nothing. Tells you whether a jump is needed at all.
        inline Result roll(float cx, Params const& pr, Box const& b, std::vector<Obstacle> const& obs,
                           float untilX, bool wantTrail) {
            return run(cx, pr, b, obs, untilX, false, wantTrail);
        }

        // Sweep start positions in [fromX, toX] and return the widest stretch where the
        // jump is safe. `center` is the best place to press jump.
        inline Window findWindow(float fromX, float toX, float stepX, Params const& pr, Box const& b,
                                 std::vector<Obstacle> const& obs, float clearX) {
            Window best;
            float bestWidth = -1.f;
            bool inRun = false;
            float runStart = 0.f, runEnd = 0.f;

            auto closeRun = [&]() {
                if (!inRun) return;
                float width = runEnd - runStart;
                if (width >= bestWidth) {
                    bestWidth = width;
                    best.found = true;
                    best.lo = runStart;
                    best.hi = runEnd;
                    best.center = (runStart + runEnd) * 0.5f;
                }
                inRun = false;
            };

            for (float x = fromX; x <= toX + 0.001f; x += stepX) {
                if (jump(x, pr, b, obs, clearX, false).safe) {
                    if (!inRun) { inRun = true; runStart = x; }
                    runEnd = x;
                } else {
                    closeRun();
                }
            }
            closeRun();
            return best;
        }
    }
}
