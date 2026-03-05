import asyncio

# CONCURRENCY: Multiple tasks running at the same time, but not necessarily simultaneously.
async def cook_pasta():
    print("Cooking pasta...")
    await asyncio.sleep(2)  # While waiting, go do something else
    print("Pasta is ready!")

async def boil_water():
    print("Boiling water...")
    await asyncio.sleep(3)  # Simulate time taken to boil water
    print("Water is boiling!")

async def main():
    # Run both tasks concurrently
    await asyncio.gather(cook_pasta(), boil_water())

asyncio.run(main())
